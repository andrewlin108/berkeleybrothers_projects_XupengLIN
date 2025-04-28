import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import time
import os
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set the risk-free rate
r = 0.03


def simulate_paths(S, r, sigma, T, M, N, seed=42):
    """Simulate asset price paths with optional seed parameter"""
    dt = T / N
    paths = np.zeros((M, N + 1))
    paths[:, 0] = S

    # Generate random paths with optional seed for reproducibility
    np.random.seed(seed)
    Z = np.random.standard_normal((M, N))
    for t in range(1, N + 1):
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1])

    return paths


def lsm_american_option_price(S, K, T, r, sigma, is_call, M=1000, N=50, seed=42, min_itm_paths=5):
    """Price American option using Least Squares Monte Carlo method with parameters"""
    # Simulate paths
    paths = simulate_paths(S, r, sigma, T, M, N, seed=seed)
    dt = T / N

    # Initialize payoffs at maturity
    if is_call:
        payoffs = np.maximum(0, paths[:, -1] - K)
    else:
        payoffs = np.maximum(0, K - paths[:, -1])

    # Working backwards through time
    for t in range(N - 1, 0, -1):
        # Identify in-the-money paths
        if is_call:
            itm = paths[:, t] > K
        else:
            itm = paths[:, t] < K

        if sum(itm) > min_itm_paths:  # Need enough ITM paths for regression
            # Current stock prices for ITM options
            S_itm = paths[itm, t]
            # Future discounted cashflows for ITM options
            V_itm = payoffs[itm] * np.exp(-r * dt * (N - t))
            # Regression basis functions (polynomial)
            X = np.column_stack([np.ones(len(S_itm)), S_itm, S_itm ** 2])
            # Fit regression model
            beta, _, _, _ = np.linalg.lstsq(X, V_itm, rcond=None)
            # Expected continuation values
            C = np.dot(X, beta)
            # Immediate exercise values
            if is_call:
                exercise = np.maximum(0, S_itm - K)
            else:
                exercise = np.maximum(0, K - S_itm)

            # Exercise decision
            ex_idx = exercise > C

            # Update payoffs
            payoffs_new = np.copy(payoffs)
            ex_path_idx = np.where(itm)[0][ex_idx]
            payoffs_new[ex_path_idx] = exercise[ex_idx] * np.exp(-r * dt * t)

            # Only consider unexercised paths for future iterations
            payoffs = payoffs_new

    # Option price is the average of all discounted payoffs
    return np.mean(payoffs)


def monte_carlo_iv(row_data, price_col, r=0.03, M=1000, N=50, max_vol=1.5, min_vol=0.01, max_iter=50, seed=42):
    """Calculate implied volatility using Monte Carlo simulation"""
    try:
        # Extract row data
        S = row_data['final_underly_mean_mid']
        K = row_data['strike']
        T = row_data['t']
        price = row_data[price_col]
        is_call = row_data['is_call'] == 1

        def objective(sigma):
            model_price = lsm_american_option_price(S, K, T, r, sigma, is_call, M=M, N=N, seed=seed)
            return model_price - price

        # Find the implied volatility with customizable parameters
        iv = brentq(objective, min_vol, max_vol, maxiter=max_iter)
        return iv
    except Exception as e:
        return np.nan


def process_batch_for_iv(batch_df, price_cols, mc_params):
    """Process a batch of rows for IV calculation with multiple price columns"""
    result_dict = {col: [] for col in price_cols}
    result_dict['index'] = []

    for idx, row in batch_df.iterrows():
        result_dict['index'].append(idx)

        for col in price_cols:
            try:
                iv = monte_carlo_iv(
                    row,
                    price_col=col,
                    r=0.03,
                    M=mc_params['M'],
                    N=mc_params['N'],
                    seed=mc_params['seed'] + idx  # Vary seed by index for better randomization
                )
                result_dict[col].append(iv)
            except Exception as e:
                result_dict[col].append(np.nan)

    return result_dict


def process_batch_parallel(args):
    """Wrapper function for parallel processing"""
    batch_idx, batch_df, price_cols, mc_params = args
    return process_batch_for_iv(batch_df, price_cols, mc_params)


def process_options_data_parallel(input_file, output_file, max_rows=None, batch_size=100,
                                  mc_params=None, num_processes=None):
    """
    Process options data with Monte Carlo IV calculation using multiprocessing

    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file
        max_rows: Maximum number of rows to process (for testing)
        batch_size: Number of rows to process in each batch
        mc_params: Parameters for Monte Carlo simulation (dict)
        num_processes: Number of processes to use (default: CPU count - 1)
    """
    start_time = time.time()
    print(f"Starting option data processing with multiprocessing...")

    # Set default Monte Carlo parameters if not provided
    if mc_params is None:
        mc_params = {
            'M': 1000,  # Number of paths
            'N': 50,  # Number of time steps
            'seed': 42
        }

    # Set number of processes
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU for system processes

    print(f"Using {num_processes} processes")

    # Read the parquet file
    df = pd.read_parquet(input_file, engine='pyarrow')
    print(f"Loaded data with {len(df)} rows")

    # Sort and prepare the data
    df = df.sort_values(by=['minute_str', 'strike']).reset_index(drop=True)
    df['t'] = df['maturity'] / 365
    df['t'] = df['t'].astype(float)

    # Get final_underly_mean_mid column
    mean_underlying_by_minute = df.groupby('minute_str')['underly_mean_mid'].mean().reset_index()
    mean_underlying_by_minute = mean_underlying_by_minute.rename(columns={'underly_mean_mid': 'final_underly_mean_mid'})
    df = df.merge(mean_underlying_by_minute, on='minute_str', how='left')

    # Limit rows for testing if specified
    if max_rows is not None:
        df = df[:max_rows]
        print(f"Limited to {len(df)} rows for testing")

    # Initialize columns for volatilities
    price_cols = ['mean_bid', 'mean_ask', 'mean_mid']
    for col in price_cols:
        vol_col = f'{col[5:]}_vol'
        df[vol_col] = np.nan

    # Prepare batch processing
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size
    batches = [(i, df.iloc[i * batch_size:min((i + 1) * batch_size, total_rows)].copy(),
                price_cols, mc_params)
               for i in range(num_batches)]

    # Process batches in parallel
    start_time_processing = time.time()
    results = []

    print(f"Processing {num_batches} batches in parallel with {num_processes} processes")

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all batches to the executor
        future_to_batch = {executor.submit(process_batch_parallel, batch_args): i
                           for i, batch_args in enumerate(batches)}

        # Track progress with tqdm
        with tqdm(total=len(batches)) as progress_bar:
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    # Get results and collect them
                    result = future.result()
                    results.append(result)

                    # Update progress
                    progress_bar.update(1)

                    # Calculate and display metrics
                    processed_rows = min((batch_idx + 1) * batch_size, total_rows)
                    elapsed_time = time.time() - start_time_processing
                    rows_per_second = processed_rows / elapsed_time if elapsed_time > 0 else 0
                    estimated_total_time = (elapsed_time / processed_rows) * total_rows if processed_rows > 0 else 0
                    remaining_time = max(0, estimated_total_time - elapsed_time)

                    # Display metrics every 10 batches or when processing is complete
                    if batch_idx % 10 == 0 or batch_idx == len(batches) - 1:
                        print(f"\rProcessed {processed_rows}/{total_rows} rows | "
                              f"Speed: {rows_per_second:.2f} rows/sec | "
                              f"Elapsed: {elapsed_time / 60:.2f} min | "
                              f"Remaining: {remaining_time / 60:.2f} min")

                    # # Save intermediate results periodically (commented out for speed)
                    # if batch_idx % 50 == 0 and batch_idx > 0:
                    #     # Apply all results gathered so far
                    #     for result in results:
                    #         for i, idx in enumerate(result['index']):
                    #             for col in price_cols:
                    #                 vol_col = f'{col[5:]}_vol'
                    #                 df.loc[idx, vol_col] = result[col][i]
                    #
                    #     intermediate_file = f"{output_file.replace('.parquet', '')}_intermediate_{batch_idx}.parquet"
                    #     df.to_parquet(intermediate_file, engine='pyarrow')
                    #     print(f"Saved intermediate results to {intermediate_file}")

                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")

    # Apply all results to the dataframe
    print("Applying results to dataframe...")
    for result in results:
        for i, idx in enumerate(result['index']):
            for col in price_cols:
                vol_col = f'{col[5:]}_vol'
                df.loc[idx, vol_col] = result[col][i]

    # Save the final results
    print(f"Saving results to {output_file}...")
    df.to_parquet(output_file, engine='pyarrow')

    total_time = time.time() - start_time
    print(f"Processing complete! Total time: {total_time / 60:.2f} minutes")
    print(f"Results saved to {output_file}")

    return df


if __name__ == "__main__":
    # Configuration
    underlying = 'CF'
    input_file = f'opt_test_data/{underlying}_options_data.parquet'
    output_file = f'opt_test_data/monte_carlo_results_{underlying}.parquet'

    # Set up Monte Carlo parameters
    mc_params = {
        'M': 1000,  # Number of paths
        'N': 50,  # Number of time steps
        'seed': 42
    }

    # Process data with parallel processing
    processed_df = process_options_data_parallel(
        input_file,
        output_file,
        max_rows=1000,  # Set to None to process the entire file
        batch_size=200,  # Smaller batch size for better load balancing
        mc_params=mc_params,
        num_processes=5  # Auto-detect CPU count
    )

    print(f"Monte Carlo processing complete! Results saved to {output_file}")