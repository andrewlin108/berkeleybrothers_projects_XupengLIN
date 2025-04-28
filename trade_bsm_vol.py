import pandas as pd
import numpy as np
from scipy.optimize import brentq, minimize_scalar
import time
from scipy.stats import norm
from scipy import optimize

from copy import deepcopy

from tqdm import tqdm

import numpy as np
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta

# Risk-free rate 和 underlying price 可以设为实际值或估计值
r = 0.03  # risk-free rate

from scipy.stats import norm
from scipy.optimize import newton
from scipy.optimize import curve_fit


import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

import plotly.graph_objs as go
import plotly.offline as pyo

def plot_pnl_and_risk(df, save_html_path, title='PnL and Risk Overview'):
    color_list = [
        'pink', 'gold', 'red', 'blue', 'slategray', 'green', 'Olive', 'DarkOrange', 'DeepSkyBlue', 'purple',
        'Black', 'LimeGreen', 'orange', 'DarkSlateBlue', 'Maroon', 'yellow', 'salmon', 'mediumturquoise',
        'gray', 'brown', 'LightPink', 'OliveDrab', 'cyan', 'navy', 'plum', 'teal', 'indigo', 'turquoise',
        'lightgreen', 'darkred', 'peru', 'firebrick', 'lightseagreen', 'mediumvioletred', 'royalblue',
        'crimson', 'khaki', 'chartreuse', 'mediumblue', 'orangered', 'seagreen', 'darkmagenta', 'goldenrod',
        'aquamarine', 'cornflowerblue', 'deeppink', 'tomato', 'darkorange', 'lightcoral', 'steelblue',
        'mediumorchid', 'springgreen', 'violet', 'dodgerblue'
    ]


    # Make sure 'time' column is string type
    if not pd.api.types.is_string_dtype(df['time']):
        df['time'] = df['time'].astype(str)
    df['time'] = df['time'].apply(lambda x: 'D' + str(x))

    trace = []
    color_idx = 0

    # Define primary y-axis fields
    primary_fields = [
        'total_pnl', 'call_pnl', 'put_pnl',
        'total_cum_pnl','call_cum_pnl', 'put_cum_pnl',
        'call_vega_cum_pnl','put_vega_cum_pnl','total_vega_cum_pnl'

    ]

    # Define secondary y-axis fields (right side)
    secondary_fields = [
        'underlying_price'
    ]

    # Plot primary fields
    for field in primary_fields:
        if field in df.columns:
            trace.append(go.Scatter(
                x=df['time'],
                y=df[field],
                mode='lines',
                name=field,
                line=dict(color=color_list[color_idx % len(color_list)], width=3),
                yaxis='y1'
            ))
            color_idx += 1

    # Plot secondary fields
    for field in secondary_fields:
        if field in df.columns:
            trace.append(go.Scatter(
                x=df['time'],
                y=df[field],
                mode='lines',
                name=field + '_R',
                line=dict(color=color_list[color_idx % len(color_list)], width=3),
                yaxis='y2'
            ))
            color_idx += 1

    # Layout settings
    layout = go.Layout(
        title=dict(text=title),
        showlegend=True,
        xaxis=dict(title='Time'),
        yaxis=dict(title='Primary Axis (PnL and Risk Metrics)'),
        yaxis2=dict(
            title='y2',
            overlaying='y',
            side='right'
        )
    )

    fig = go.Figure(data=trace, layout=layout)
    pyo.plot(fig, filename=save_html_path, validate=False, auto_open=False)
    print(f"✅ Plot saved to: {save_html_path}")

def vol_curve_trading_opportunities(df_latest_price, vol_threshold=0.02, position_size=1):
    """
    Check for trading opportunities based on fitted vol vs final vol.
    暂时默认触发就全平仓
    Args:
        df_latest_price: DataFrame containing option data
        vol_threshold: Threshold for volatility difference to trigger a trade
        position_size: Number of contracts to trade when signal is triggered

    Returns:
        Updated DataFrame with modified call_qty and put_qty values
    """
    # Make a copy to avoid modifying the original
    df = df_latest_price.copy()

    # Process each strike price
    for idx, row in df.iterrows():
        strike = row['strike']
        underlying = row['final_underly_mean_mid']

        # Determine if the option is OTM
        is_call_otm = strike > underlying
        is_put_otm = strike < underlying

        # Long signal: fitted_vol > final_vol + threshold
        long_signal = row['fitted_vol'] > row['final_vol'] + vol_threshold

        # Short signal: fitted_vol < final_vol - threshold
        short_signal = row['fitted_vol'] < row['final_vol'] - vol_threshold

        # Close signal: fitted_vol ≈ final_vol (in between thresholds)
        close_signal = not (long_signal or short_signal)

        # Update call position
        if is_call_otm:  # Only trade OTM calls
            if long_signal and row['call_qty'] == 0:
                # Go long on call
                df.at[idx, 'call_qty'] += position_size
                print(
                    f"OPEN LONG CALL at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")
            elif short_signal and row['call_qty'] == 0:
                # Go short on call
                df.at[idx, 'call_qty'] -= position_size
                print(
                    f"OPEN SHORT CALL at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")


        # Update put position
        if is_put_otm:  # Only trade OTM puts
            if long_signal and row['put_qty'] == 0:
                # Go long on put
                df.at[idx, 'put_qty'] += position_size
                print(
                    f"OPEN LONG PUT at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")
            elif short_signal and row['put_qty'] == 0:
                # Go short on put
                df.at[idx, 'put_qty'] -= position_size
                print(
                    f"OPEN SHORT PUT at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")


        if row['call_qty'] > 0 and row['fitted_vol'] <= row['final_vol']:
            # Close position
            print(
                f"CLOSE Long CALL at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")
            df.at[idx, 'call_qty'] = 0
        elif row['call_qty'] < 0 and row['fitted_vol'] >= row['final_vol']:
            # Close position
            print(
                f"CLOSE short CALL at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")
            df.at[idx, 'call_qty'] = 0

        if row['put_qty'] > 0 and row['fitted_vol'] <= row['final_vol']:
            # Close position
            print(
                f"CLOSE Long PUT at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")
            df.at[idx, 'put_qty'] = 0
        elif row['put_qty'] < 0 and row['fitted_vol'] >= row['final_vol']:
            # Close position
            print(
                f"CLOSE short PUT at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")
            df.at[idx, 'put_qty'] = 0

    return df



def calculate_vega_pnl(current_df, previous_df=None):
    """
    Calculate P&L based on volatility changes.

    Args:
        current_df: Current DataFrame with option data
        previous_df: Previous DataFrame for comparison (if None, P&L is 0)

    Returns:
        DataFrame with added P&L columns and total P&L
    """
    df = current_df.copy()

    # Initialize P&L columns if they don't exist
    if 'call_vega_pnl' not in df.columns:
        df['call_vega_pnl'] = 0.0
    if 'put_vega_pnl' not in df.columns:
        df['put_vega_pnl'] = 0.0
    if 'total_vega_pnl' not in df.columns:
        df['total_vega_pnl'] = 0.0

    total_vega_pnl = 0.0
    total_call_vega_pnl=0.0
    total_put_vega_pnl=0.0

    # If there's no previous data, return zeros
    if previous_df is None:
        return df, total_vega_pnl,total_call_vega_pnl,total_put_vega_pnl

    # Ensure both dataframes have the same strikes
    common_strikes = set(df['strike']).intersection(set(previous_df['strike']))

    for strike in common_strikes:
        curr_row = df[df['strike'] == strike].iloc[0]
        prev_row = previous_df[previous_df['strike'] == strike].iloc[0]

        idx = df[df['strike'] == strike].index[0]

        # Calculate call P&L based on vega and vol change
        if prev_row['call_qty'] != 0:
            call_vol_change = curr_row['call_vol'] - prev_row['call_vol']



            # P&L = quantity * vega * vol change * 100 (as vega is per 1% change)
            call_pnl = prev_row['call_qty'] * curr_row['call_vega'] * (call_vol_change * 100)
            if np.isnan(call_pnl):
                call_pnl = 0

            df.at[idx, 'call_pnl'] = call_pnl
            total_vega_pnl += call_pnl
            total_call_vega_pnl+=call_pnl

        # Calculate put P&L based on vega and vol change
        if prev_row['put_qty'] != 0:
            put_vol_change = curr_row['put_vol'] - prev_row['put_vol']

            # P&L = quantity * vega * vol change * 100 (as vega is per 1% change)
            put_pnl = prev_row['put_qty'] * curr_row['put_vega'] * (put_vol_change * 100)
            if np.isnan(put_pnl):
                put_pnl = 0

            df.at[idx, 'put_pnl'] = put_pnl
            total_vega_pnl += put_pnl
            total_put_vega_pnl += put_pnl
        if np.isnan(total_put_vega_pnl) or np.isnan(total_call_vega_pnl):
            print('check nan pnl')
        # Calculate total P&L for this strike
        df.at[idx, 'total_vega_pnl'] = df.at[idx, 'call_vega_pnl'] + df.at[idx, 'put_vega_pnl']

    return df, total_vega_pnl,total_call_vega_pnl,total_put_vega_pnl

def calculate_pnl(current_df, previous_df=None):
    """
    Calculate P&L based on volatility changes.

    Args:
        current_df: Current DataFrame with option data
        previous_df: Previous DataFrame for comparison (if None, P&L is 0)

    Returns:
        DataFrame with added P&L columns and total P&L
    """
    df = current_df.copy()

    # Initialize P&L columns if they don't exist
    if 'call_pnl' not in df.columns:
        df['call_pnl'] = 0.0
    if 'put_pnl' not in df.columns:
        df['put_pnl'] = 0.0
    if 'total_pnl' not in df.columns:
        df['total_pnl'] = 0.0

    total_pnl = 0.0
    total_call_pnl=0.0
    total_put_pnl=0.0

    # If there's no previous data, return zeros
    if previous_df is None:
        return df, total_pnl,total_call_pnl,total_put_pnl

    # Ensure both dataframes have the same strikes
    common_strikes = set(df['strike']).intersection(set(previous_df['strike']))

    for strike in common_strikes:
        curr_row = df[df['strike'] == strike].iloc[0]
        prev_row = previous_df[previous_df['strike'] == strike].iloc[0]

        idx = df[df['strike'] == strike].index[0]

        # Calculate call P&L based on vega and vol change
        if prev_row['call_qty'] != 0:
            call_price_change = curr_row['call_mean_mid'] - prev_row['call_mean_mid']
            # P&L = quantity * vega * vol change * 100 (as vega is per 1% change)
            call_pnl = prev_row['call_qty'] * call_price_change
            df.at[idx, 'call_pnl'] = call_pnl
            total_pnl += call_pnl
            total_call_pnl+=call_pnl

        # Calculate put P&L based on vega and vol change
        if prev_row['put_qty'] != 0:
            put_price_change = curr_row['put_mean_mid'] - prev_row['put_mean_mid']
            # P&L = quantity * vega * vol change * 100 (as vega is per 1% change)
            put_pnl = prev_row['put_qty'] * put_price_change
            df.at[idx, 'put_pnl'] = put_pnl
            total_pnl += put_pnl
            total_put_pnl += put_pnl

        # Calculate total P&L for this strike
        df.at[idx, 'total_pnl'] = df.at[idx, 'call_pnl'] + df.at[idx, 'put_pnl']

    return df, total_pnl,total_call_pnl,total_put_pnl


def plot_volatility_curves(df, time_stamp, save_dir=None,if_show=False):
    """
    Plot volatility curves including fitted vol, call/put vols (mid, bid, ask)

    Args:
        df: DataFrame containing volatility data
        time_stamp: Timestamp of the data (used in title and filename)
        save_dir: Directory to save the plot (optional)
    """


    plt.figure(figsize=(12, 8))

    # Sort by strike to ensure smooth line
    df = df.sort_values('strike')

    # Plot fitted vol as a connected line
    plt.plot(df['strike'], df['fitted_vol'], 'r-', linewidth=2, label='Fitted Vol')

    plt.scatter(df['strike'], df['final_vol'], marker='*', color='purple', s=100, alpha=0.7, label='Final Vol')

    # Plot call volatilities
    plt.scatter(df['strike'], df['call_vol'], marker='o', color='blue', alpha=0.7, label='Call Mid Vol')
    plt.scatter(df['strike'], df['call_bid_vol'], marker='^', color='lightblue', alpha=0.5, label='Call Bid Vol')
    plt.scatter(df['strike'], df['call_ask_vol'], marker='v', color='darkblue', alpha=0.5, label='Call Ask Vol')

    # Plot put volatilities
    plt.scatter(df['strike'], df['put_vol'], marker='s', color='green', alpha=0.7, label='Put Mid Vol')
    plt.scatter(df['strike'], df['put_bid_vol'], marker='^', color='lightgreen', alpha=0.5, label='Put Bid Vol')
    plt.scatter(df['strike'], df['put_ask_vol'], marker='v', color='darkgreen', alpha=0.5, label='Put Ask Vol')

    # Add vertical line at underlying price
    underlying_price = df['final_underly_mean_mid'].iloc[0]
    plt.axvline(x=underlying_price, color='grey', linestyle='--', label=f'Underlying ({underlying_price:.2f})')


    # Format the plot
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title(f'Volatility Smile at {time_stamp}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')

    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))

    # Add time to maturity and date info
    t_value = df['t'].iloc[0]
    days_to_maturity = int(df['maturity'].iloc[0])
    plt.figtext(0.02, 0.02, f'Time to Maturity: {t_value:.4f} years ({days_to_maturity} days)',
                ha='left', fontsize=10)

    # Tighten layout
    plt.tight_layout()

    if if_show:
        plt.show()
    # Save if directory is provided
    if save_dir:
        # Convert timestamp to string format for filename
        if hasattr(time_stamp, 'strftime'):
            time_str = time_stamp.strftime('%Y%m%d_%H%M%S')
        else:
            time_str = str(time_stamp).replace(' ', '_').replace(':', '').replace('-', '')

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save plot
        file_path = os.path.join(save_dir, f'vol_smile_{time_str}.png')
        plt.savefig(file_path, dpi=300)
        print(f"Saved plot to {file_path}")

    return plt.gcf()  # Return the figure object if needed


def vol_fitting(df_cut, time_stamp):
    global time_stamp_list, ATM_vol, Skew_list, Kp_list, SKP_list, Kc_list, SKC_list
    # ############
    # df_cut = pd.concat([df_cut, df_cut[:8]])
    # df_cut['ispot'] = 4.92531
    # df_cut['t'] = 0.0326531
    # df_cut['final_vol'] = [0.756184, 0.699271, 0.650223, 0.594569, 0.543918, 0.494269, 0.445531, 0.414917, 0.373983,
    #                        0.338447, 0.298127, 0.260829, 0.237162, 0.207897, 0.190118, 0.180178, 0.180662, 0.222172,
    #                        0.282631, 0.362166, 0.415824, 0.481131]
    # df_cut['strike_price'] = [3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, 5.25,
    #                           5.5, 5.75, 6, 6.25]
    ###############
    global Fwd, t

    df_cut = df_cut.reset_index(drop=True)
    df_original = deepcopy(df_cut)
    x_original = np.array(df_original['strike'])
    r = 0.03
    t = df_cut['t'].values[0]
    # Fwd = df_cut['final_underly_mean_mid'].values[0] * np.exp(-r * t)
    Fwd = df_cut['final_underly_mean_mid'].values[0]
    x_list = []
    y_list = []
    df_cut = df_cut.dropna(subset=['final_vol'])
    df_cut = df_cut.reset_index(drop=True)
    d1 = (np.log(df_cut["final_underly_mean_mid"].astype(float) / df_cut['strike']) + (r + df_cut['final_vol'] ** 2 / 2) *
              df_cut['t']) / (df_cut['final_vol'] * np.sqrt(df_cut["t"].astype(float)))
    d1=d1.astype(float)


    df_cut['vega'] = df_cut["final_underly_mean_mid"].astype(float) * np.sqrt(df_cut["t"].astype(float)) * norm.pdf(d1)
    max_vega = df_cut['vega'].max()
    df_cut['vega'] = df_cut['vega']/max_vega

    for i in range(0, len(df_cut)):
        times = max(int(round(df_cut.loc[i]['vega']*100)), 10)
        x_list = x_list + [df_cut.loc[i]['strike']] * times
        y_list = y_list + [df_cut.loc[i]['final_vol']] * times
    x = np.array(x_list)
    y = np.array(y_list)

    ###########
    # 非线性最小二乘法拟合
    try:
        popt, pcov = curve_fit(piecewise_linear, x, y, maxfev=1000)
        # 拟合系数存储csv
        # 获取popt里面是拟合系数

        # 计算original strike的system_vol
        ATM = popt[0]
        Skew = popt[1]
        Kp = popt[2]
        SKP = popt[3]
        Kc = popt[4]
        SKC = popt[5]
        yvals = piecewise_linear(x_original, ATM, Skew, Kp, SKP, Kc, SKC)  # 拟合y值
        # print(popt, yvals)
        df_original['fitted_vol'] = np.array(yvals)
        # 如果fit出来的波动率有负数，整个扔掉，不要fit
        if len(df_original[df_original['fitted_vol'] < 0]) > 0:
            time_stamp_list.append(time_stamp)
            ATM_vol.append(np.nan)
            Skew_list.append(np.nan)
            Kp_list.append(np.nan)
            SKP_list.append(np.nan)
            Kc_list.append(np.nan)
            SKC_list.append(np.nan)
            # df_original['fitted_vol'] = np.nan
            # df_original['system_vol'] = np.nan
            df_original['fitted_vol'] = df_original['final_vol']
            df_original['vol_flag'] = 1
        else:
            time_stamp_list.append(time_stamp)
            ATM_vol.append(ATM)
            Skew_list.append(Skew)
            Kp_list.append(Kp)
            SKP_list.append(SKP)
            Kc_list.append(Kc)
            SKC_list.append(SKC)
            # if str(time_stamp)[14:16] + str(time_stamp)[17:19] == '3000':
            # plotting_html(df_original, time_stamp, expiry_date)
    except:
        print('Optimal parameters not found: Number of calls to function has reached maxfev = 1200')
        time_stamp_list.append(time_stamp)
        ATM_vol.append(np.nan)
        Skew_list.append(np.nan)
        Kp_list.append(np.nan)
        SKP_list.append(np.nan)
        Kc_list.append(np.nan)
        SKC_list.append(np.nan)
        # df_original['fitted_vol'] = np.nan
        # df_original['system_vol'] = np.nan
        df_original['fitted_vol'] = df_original['final_vol']
        df_original['vol_flag'] = 1
    # 修正fitted_vol为system_vol
    # df_original = get_system_vol(df_original)
    # system_vol = df_original[['call_instr_id', 'bid_vol_diff', 'ask_vol_diff', 'bid_vol', 'ask_vol', 'fitted_vol',
    #                           'system_vol', 'vol_flag']]

    return df_original

def calc_implied_vols(call_put: str, df: pd.DataFrame, r: float = 0.03, initial_guess=0.2, tol=1e-5):
    S = df["ispot"].values
    K = df["strike"].values
    t = df["t"].values
    bid = df[f"{call_put}_bid"].values
    ask = df[f"{call_put}_ask"].values
    mid = df[f"{call_put}_market_price"].values
    vol = np.array([initial_guess] * len(df))

    def bs_price(v, cp, S, K, t, r):
        d1 = (np.log(S / K) + (r + 0.5 * v**2) * t) / (v * np.sqrt(t))
        d2 = d1 - v * np.sqrt(t)
        if cp == "call":
            return norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * t)
        else:
            return norm.cdf(-d2) * K * np.exp(-r * t) - norm.cdf(-d1) * S

    def iv_solver(v, price, cp, S, K, t, r):
        return bs_price(v, cp, S, K, t, r) - price

    df[f"{call_put}_bid_vol"] = [np.nan] * len(df)
    df[f"{call_put}_ask_vol"] = [np.nan] * len(df)
    df[f"{call_put}_vol"] = [np.nan] * len(df)

    for i in range(len(df)):
        for ptype, pvalue, label in zip(
            ["bid", "ask", "market_price"], [bid[i], ask[i], mid[i]],
            ["bid_vol", "ask_vol", "vol"]
        ):
            if pvalue > 0 and S[i] > 0 and K[i] > 0 and t[i] > 0:
                try:
                    solved_vol = newton(
                        iv_solver, initial_guess, args=(pvalue, call_put, S[i], K[i], t[i], r),
                        tol=tol, maxiter=100
                    )
                    df.loc[df['strike']==K, f"{call_put}_{label}"] = solved_vol
                except:
                    pass
    return df




def calc_bs_fair_and_greeks(call_put: str, df: pd.DataFrame, r: float = 0.03):
    S = df["ispot"].values
    K = df["strike"].values
    t = df["t"].values
    vol = df[f"{call_put}_vol"].values  # using mid vol
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)

    df[f"{call_put}_fair"] = (
        norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * t)
        if call_put == "call"
        else norm.cdf(-d2) * K * np.exp(-r * t) - norm.cdf(-d1) * S
    )

    df[f"{call_put}_delta"] = (
        norm.cdf(d1) if call_put == "call" else norm.cdf(d1) - 1.0
    )

    theta = (
        -((S * norm.pdf(d1) * vol) / (2 * np.sqrt(t))) -
        (r * K * np.exp(-r * t) * norm.cdf(d2))
        if call_put == "call"
        else
        -((S * norm.pdf(d1) * vol) / (2 * np.sqrt(t))) +
        (r * K * np.exp(-r * t) * norm.cdf(-d2))
    )
    df[f"{call_put}_theta"] = theta / 365

    rho = (
        K * t * np.exp(-r * t) * norm.cdf(d2)
        if call_put == "call"
        else -K * t * np.exp(-r * t) * norm.cdf(-d2)
    )
    df[f"{call_put}_rho"] = rho / 100

    df[f"{call_put}_gamma"] = norm.pdf(d1) / (S * vol * np.sqrt(t))
    df[f"{call_put}_vega"] = S * norm.pdf(d1) * np.sqrt(t) / 100

    return df


# 计算隐含波动率
def compute_iv_euro(row):
    try:
        S = row['underly_mean_mid']             # 标的价格
        K = row['strike']                       # 行权价
        T = row['maturity'] / 365               # 到期时间 (按日转年)
        price = row['mean_mid']                 # 使用 mid 价
        flag = 'c' if row['is_call'] == 1 else 'p'

        # 用 Black-Scholes 模型近似美式期权
        iv = implied_volatility(price, S, K, T, r, flag)
        return iv
    except:
        return np.nan


# Vol smile model
def piecewise_linear(x, ATM, Skew, Kp, SKP, Kc, SKC):
    log_term = np.log(Fwd / x) / np.sqrt(t)
    left = ATM + Skew * log_term + 0.5 * Kp * log_term**2 + (5 / 3) * SKP * log_term**3
    right = ATM + Skew * log_term + 0.5 * Kc * log_term**2 - (5 / 3) * SKC * log_term**3
    return np.where(x <= Fwd, left, right)

if __name__ == '__main__':
    underlying = 'CF'
    # underlying = 'OI'
    # underlying = 'RM'
    # underlying = 'SR'
    file_name = 'opt_test_data/{}_options_data.parquet'.format(underlying)

    # file_name = 'opt_test_data/monte_carlo_results_CF.parquet'.format(underlying)
    # 读取文件
    df = pd.read_parquet(file_name, engine='pyarrow')

    df = df.sort_values(by=['minute_str', 'strike']).reset_index(drop=True)
    df['t']=df['maturity']/365
    df['t']=df['t'].astype(float)
    # df[:1000].to_csv('sample_data.csv')

    strike_counts_per_minute = df.groupby(['minute_str', 'strike']).size().reset_index(name='count')
    if sum(strike_counts_per_minute['count']!=2)>0:
        print('call put存在无法配对的情况,占比为{}'.format(sum(strike_counts_per_minute['count']!=2)/len(strike_counts_per_minute)))

    maturity_diff_set=set(df['maturity'].diff())
    print(maturity_diff_set)
    #maturity为剩余自然日

    #差不多20个交易日
    # df=df[:100000]

    # 先跑通试试
    # df=df[:1000]

    # df_filtered = df[:1000]
    # t0=time.time()
    # df_filtered['iv_euro'] = df_filtered.apply(compute_iv_euro, axis=1)
    #
    # t1=time.time()
    # print('time spent:{}'.format(t1-t0))



    t2=time.time()

    mean_underlying_by_minute = df.groupby('minute_str')['underly_mean_mid'].mean().reset_index()
    mean_underlying_by_minute = mean_underlying_by_minute.rename(columns={'underly_mean_mid': 'final_underly_mean_mid'})

    df = df.merge(mean_underlying_by_minute, on='minute_str', how='left')

    df['minute_str'] = pd.to_datetime(df['minute_str'])
    df = df.sort_values(by='minute_str')

    # Columns we want to track for each call/put leg
    base_columns = ['mean_ask', 'mean_bid', 'cumcashvol', 'mean_mid', 'cumvolume', 'openinterest', 'twap']

    # Initialize empty DataFrame for tracking the latest option data
    extra_columns = ['maturity','t', 'final_underly_mean_mid']
    df_latest_price = pd.DataFrame(columns=['strike'] +
                                           [f'call_{col}' for col in base_columns] + ['call_time'] +
                                           [f'put_{col}' for col in base_columns] + ['put_time'] +
                                           extra_columns)
    df_latest_price['call_qty']=0
    df_latest_price['put_qty']=0
    df_latest_price['call_pnl']=0
    df_latest_price['put_pnl']=0
    df_latest_price['total_pnl']=0
    df_latest_price['call_vega_pnl']=0
    df_latest_price['put_vega_pnl']=0
    df_latest_price['total_vega_pnl']=0
    greek_suffixes = ['_vol', '_bid_vol', '_ask_vol', '_delta', '_theta', '_rho', '_gamma', '_vega', '_fair']
    for col in greek_suffixes:
        df_latest_price[f"call{col}"] = np.nan
        df_latest_price[f"put{col}"] = np.nan


    # Set index to strike for easy updating
    # df_latest_price.set_index('strike', inplace=True)

    global time_stamp_list, ATM_vol, Skew_list, Kp_list, SKP_list, Kc_list, SKC_list
    time_stamp_list = []
    ATM_vol = []
    Skew_list = []
    Kp_list = []
    SKP_list = []
    Kc_list = []
    SKC_list = []

    # Process updates minute by minute
    # Step 2: Loop through time

    df_all_minute_price=pd.DataFrame()
    df_history = {}
    df_minute_pnl=pd.DataFrame()

    df['underlying_instr']=df['Instrument'].apply(lambda x:x[:6])
    current_instr = df['underlying_instr'].iloc[0]
    previous_instr = df['underlying_instr'].iloc[0]
    if_new_undelrying_instr=False
    num_minutes=len(df['minute_str'].unique())
    for minute, group in tqdm(df.groupby('minute_str'), total=num_minutes,
                              desc="Processing minutes", unit="minute"):
        print(f"Processing minute: {minute}")
        current_instr = group['underlying_instr'].iloc[0]
        if current_instr!=previous_instr:
            print('underlying instr change')
            #clear current price info
            df_latest_price=df_latest_price[:0]
            underlying_hedge_qty=0
            #TODO: clear position and calculate the cose
            if_new_undelrying_instr=True

        # 更新每一个价格，波动率和greeks
        for _, row in group.iterrows():
            strike = row['strike']
            is_call = row['is_call'] == 1
            prefix = 'call_' if is_call else 'put_'
            side = prefix[:-1]  # 'call' or 'put'

            # Create the update dictionary for this row
            update_data = {f"{prefix}{col}": row[col] for col in base_columns}
            update_data[f"{prefix}time"] = row['minute_str']
            update_data['maturity'] = row['maturity']
            update_data['t'] = row['t']
            update_data['final_underly_mean_mid'] = row['final_underly_mean_mid']

            if strike not in df_latest_price['strike'].values:
                empty_row = pd.Series({col: np.nan for col in df_latest_price.columns})
                empty_row['strike'] = strike
                empty_row['call_qty'] = 0
                empty_row['put_qty'] = 0
                empty_row['call_pnl'] = 0
                empty_row['put_pnl'] = 0
                df_latest_price = pd.concat([df_latest_price, empty_row.to_frame().T])

            for k, v in update_data.items():
                df_latest_price.loc[df_latest_price['strike'] == strike, k] = v
                # df_latest_price.at[strike, k] = v

            # Step 3: Greeks and vol calculation
            try:
                temp_df = pd.DataFrame({
                    "ispot": [row['final_underly_mean_mid']],
                    "strike": [row['strike']],
                    "interest_rate": [r],
                    "t": [row['t']],
                    f"{side}_market_price": [row['mean_mid']],
                    f"{side}_bid": [row['mean_bid']],
                    f"{side}_ask": [row['mean_ask']],
                })

                # Vol and Greeks calculation
                temp_df = calc_implied_vols(side, temp_df, r=r)
                temp_df["mid_vol"] = temp_df[f"{side}_vol"]
                temp_df = calc_bs_fair_and_greeks(side, temp_df, r=r)

                for g in greek_suffixes:
                    col_name = f"{side}{g}"
                    if col_name in temp_df.columns:
                        # df_latest_price.at[strike, col_name] = temp_df[col_name].iloc[0]
                        df_latest_price.loc[df_latest_price['strike']==strike, col_name] = temp_df[col_name].iloc[0]

            except Exception as e:
                print(f"Vol/Greek calc failed at strike {strike}: {e}")

        df_latest_price = df_latest_price.sort_values(by=['strike']).reset_index(drop=True)
        # df_latest_price['strike_price']=df_latest_price.index
        df_latest_price['final_vol'] = np.where(
            df_latest_price['strike'] >= df_latest_price['final_underly_mean_mid'],
            df_latest_price['call_vol'],  # call mid vol placeholder
            df_latest_price['put_vol']  # put mid vol placeholder
        )
        underlying_price=df_latest_price['final_underly_mean_mid'].iloc[0]
        # fit the vol curve
        df_latest_price=vol_fitting(df_latest_price,minute)

        # check vol curve and see if any pattern exists
        # plot_volatility_curves(df_latest_price,minute,if_show=True)

        # Find trading opportunities and update quantities
        df_latest_price = vol_curve_trading_opportunities(df_latest_price, vol_threshold=0.001,
                                                      position_size=100)

        # risk control, check risk and do hedging


        # Store the current state in history
        df_history[minute] = df_latest_price.copy()

        # If we have previous data, calculate P&L
        if len(df_history) > 1 and not if_new_undelrying_instr:
            prev_timestamp = list(df_history.keys())[-2]
            df_latest_price, minute_vega_pnl,call_vega_pnl,put_vega_pnl = calculate_vega_pnl(df_latest_price, df_history[prev_timestamp])

            df_latest_price, minute_pnl,call_pnl,put_pnl = calculate_pnl(df_latest_price, df_history[prev_timestamp])
            print(f"Minute: {minute}, P&L: {minute_pnl:.2f}")
            df_current_pnl=pd.DataFrame({'time':[minute],'underlying_price':[underlying_price],
                                         'total_pnl':[minute_pnl],'call_pnl':[call_pnl], 'put_pnl':[put_pnl],
                                         'total_vega_pnl':[minute_vega_pnl],'call_vega_pnl':[call_vega_pnl], 'put_vega_pnl':[put_vega_pnl]})
            df_minute_pnl=pd.concat([df_minute_pnl,df_current_pnl])
            print()

        previous_instr=current_instr
        if_new_undelrying_instr = False
        # print()

    #当作european option put会被高估，所以亏钱很正常
    df_minute_pnl['call_vega_cum_pnl']=df_minute_pnl['call_vega_pnl'].cumsum()
    df_minute_pnl['put_vega_cum_pnl']=df_minute_pnl['put_vega_pnl'].cumsum()
    df_minute_pnl['total_vega_cum_pnl']=df_minute_pnl['total_vega_pnl'].cumsum()

    df_minute_pnl['call_cum_pnl']=df_minute_pnl['call_pnl'].cumsum()
    df_minute_pnl['put_cum_pnl']=df_minute_pnl['put_pnl'].cumsum()
    df_minute_pnl['total_cum_pnl']=df_minute_pnl['total_pnl'].cumsum()

    os.makedirs('backtest_results_plots',exist_ok=True)
    saved_name='backtest_results_plots/{} vega PnL Plot based on BSM vol'.format(underlying)
    plot_pnl_and_risk(
        df_minute_pnl,
        save_html_path=saved_name,
        title='{} PnL and Risk Plot based on BSM vol'.format(underlying)
    )

    print()