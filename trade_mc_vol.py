import pandas as pd
import numpy as np
import time
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from py_vollib.black_scholes.implied_volatility import implied_volatility
from scipy.stats import norm
from scipy.optimize import newton
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os
import plotly.graph_objs as go
import plotly.offline as pyo


# Function to plot PnL and risk metrics
def plot_pnl_and_risk(df, save_html_path, title='PnL and Risk Overview'):
    """
    Plot PnL (Profit and Loss) and risk metrics over time using Plotly.

    Args:
        df: DataFrame containing PnL and risk metrics data.
        save_html_path: Path to save the generated HTML plot.
        title: Title of the plot.

    Returns:
        None. Saves the plot as an HTML file.
    """

    color_list = [
        'pink', 'gold', 'red', 'blue', 'slategray', 'green', 'Olive', 'DarkOrange', 'DeepSkyBlue', 'purple',
        'Black', 'LimeGreen', 'orange', 'DarkSlateBlue', 'Maroon', 'yellow', 'salmon', 'mediumturquoise',
        'gray', 'brown', 'LightPink', 'OliveDrab', 'cyan', 'navy', 'plum', 'teal', 'indigo', 'turquoise',
        'lightgreen', 'darkred', 'peru', 'firebrick', 'lightseagreen', 'mediumvioletred', 'royalblue',
        'crimson', 'khaki', 'chartreuse', 'mediumblue', 'orangered', 'seagreen', 'darkmagenta', 'goldenrod',
        'aquamarine', 'cornflowerblue', 'deeppink', 'tomato', 'darkorange', 'lightcoral', 'steelblue',
        'mediumorchid', 'springgreen', 'violet', 'dodgerblue'
    ]


    # Ensure the 'time' column is in string format for better visualization
    if not pd.api.types.is_string_dtype(df['time']):
        df['time'] = df['time'].astype(str)
    df['time'] = df['time'].apply(lambda x: 'D' + str(x))

    trace = []
    color_idx = 0

    # Define primary y-axis fields
    primary_fields = [
        'option_cum_pnl', 'call_cum_pnl', 'put_cum_pnl',
        'bid_ask_cum_fee', 'option_cum_fee', 'underlying_hedge_cum_pnl',
        'total_cum_pnl_with_fee', 'total_cum_pnl_with_hedge', 'total_cum_pnl_with_fee_and_hedge', 'delta_cum_pnl', 'gamma_cum_pnl', 'vega_cum_pnl', 'theta_cum_pnl'
    ]

    # Define secondary y-axis fields (right side)
    secondary_fields = [
        'total_option_delta', 'total_gamma_dollar',
        'total_vega','total_theta', 'underlying_price'
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

    # Create the figure and save it as an HTML file
    fig = go.Figure(data=trace, layout=layout)
    pyo.plot(fig, filename=save_html_path, validate=False, auto_open=False)
    print(f"Plot saved to: {save_html_path}")


def basic_order_risk_control(df_current,current_delta_dollar,current_gamma_dollar,delta_dollar_hedge_limit,gamma_dollar_hedge_limit):
    """
    Adjusts option orders to control delta and gamma risk exposures.
    If current gamma dollar + exceed the gamma limit，will adjust the order qty to let final gamma get close to 0.
    Reduce the order of options that has larger gamma first

    (currently just control gamma,delta and gamma control together will have more choice, need more code to clarify how to control)

    Args:
        df_current: DataFrame containing the current option data and positions.
        current_delta_dollar: Current total delta exposure in dollar terms.
        current_gamma_dollar: Current total gamma exposure in dollar terms.
        delta_dollar_hedge_limit: Maximum allowed delta exposure in dollar terms.
        gamma_dollar_hedge_limit: Maximum allowed gamma exposure in dollar terms.

    Returns:
        Updated DataFrame with adjusted orders to control delta and gamma exposures.
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df = df_current.copy()

    # Calculate delta, gamma dollar contribution for each order position

    # df['delta_dollar_contribution'] = df['volume_multiple'] * df['final_underly_mean_mid'] * (
    #         df['call_delta'] * df['call_current_order_qty'] +
    #         df['put_delta'] * df['put_current_order_qty']
    # )

    df['gamma_dollar_contribution'] = df['final_underly_mean_mid'] * df['final_underly_mean_mid'] * \
                                    df['gamma'] * (df['call_current_order_qty'] + df['put_current_order_qty']) * \
                                    df['volume_multiple'] / 100

    # current_total_delta_dolalr_contribution=df['delta_dollar_contribution'].sum()
    current_order_gamma_dolalr_contribution=df['gamma_dollar_contribution'].sum()

    #TODO: improve algorithm to try to control delta and gamma exposure together

    # If current gamma dollar + exceed the gamma limit，will adjust the order qty to let final gamma get close to 0
    if(current_gamma_dollar+current_order_gamma_dolalr_contribution>gamma_dollar_hedge_limit):
        positive_gamma_positions = df[df['gamma_dollar_contribution'] > 0].copy()
        if not positive_gamma_positions.empty:
            # Sort positions by gamma contribution in descending order
            positive_gamma_positions = positive_gamma_positions.sort_values('gamma_dollar_contribution', ascending=False)

            # Calculate the total gamma reduction needed to get close to 0
            gamma_reduction_needed = current_gamma_dollar+current_order_gamma_dolalr_contribution
            gamma_reduced = 0

            for idx, row in positive_gamma_positions.iterrows():

                orig_call_qty = df.loc[idx, 'call_current_order_qty']
                orig_put_qty = df.loc[idx, 'put_current_order_qty']

                # 计算每份合约的gamma贡献
                gamma_per_contract = row['gamma'] * row['final_underly_mean_mid'] * row['final_underly_mean_mid'] * row[
                    'volume_multiple'] / 100
                try:
                    # 调整call order（如果是多头call）
                    if orig_call_qty > 0:
                        contracts_to_reduce = min(orig_call_qty, int(gamma_reduction_needed / gamma_per_contract) + 1)
                        df.loc[idx, 'call_current_order_qty'] -= contracts_to_reduce
                        gamma_reduced += contracts_to_reduce * gamma_per_contract
                        print(f"减少 {contracts_to_reduce} 张多头call合约 (行权价: {row['strike']:.2f}) 以控制gamma")

                    # 调整put order（如果是多头put）
                    if orig_put_qty > 0:
                        contracts_to_reduce = min(orig_put_qty, int(gamma_reduction_needed / gamma_per_contract) + 1)
                        df.loc[idx, 'put_current_order_qty'] -= contracts_to_reduce
                        gamma_reduced += contracts_to_reduce * gamma_per_contract
                        print(f"减少 {contracts_to_reduce} 张多头put合约 (行权价: {row['strike']:.2f}) 以控制gamma")
                except:
                    print('please check')

                # 检查是否已经减少了足够的gamma
                if gamma_reduced >= gamma_reduction_needed:
                    print('reduce enough long gamma')
                    break


    elif(current_gamma_dollar+current_order_gamma_dolalr_contribution<-gamma_dollar_hedge_limit):
        negative_gamma_positions = df[df['gamma_dollar_contribution'] < 0].copy()
        if not negative_gamma_positions.empty:
            # Sort positions by gamma contribution in ascending order
            negative_gamma_positions = negative_gamma_positions.sort_values('gamma_dollar_contribution', ascending=True)

            # Calculate the total gamma reduction needed to get close to 0
            gamma_reduction_needed = current_gamma_dollar+current_order_gamma_dolalr_contribution
            gamma_reduced = 0

            for idx, row in negative_gamma_positions.iterrows():
                orig_call_qty = df.loc[idx, 'call_current_order_qty']
                orig_put_qty = df.loc[idx, 'put_current_order_qty']

                # 计算每份合约的gamma贡献
                gamma_per_contract = row['gamma'] * row['final_underly_mean_mid'] * row['final_underly_mean_mid'] * row[
                    'volume_multiple'] / 100
                try:
                    # 调整call order（如果是多头call）
                    if orig_call_qty < 0:
                        contracts_to_reduce = max(orig_call_qty, int(gamma_reduction_needed / gamma_per_contract) - 1)
                        df.loc[idx, 'call_current_order_qty'] -= contracts_to_reduce
                        gamma_reduced += contracts_to_reduce * gamma_per_contract
                        print(f"减少 {contracts_to_reduce} 张空头call合约 (行权价: {row['strike']:.2f}) 以控制gamma")

                    # 调整put order（如果是多头put）
                    if orig_put_qty < 0:
                        contracts_to_reduce = max(orig_put_qty, int(gamma_reduction_needed / gamma_per_contract) - 1)
                        df.loc[idx, 'put_current_order_qty'] -= contracts_to_reduce
                        gamma_reduced += contracts_to_reduce * gamma_per_contract

                        print(f"减少 {contracts_to_reduce} 张空头put合约 (行权价: {row['strike']:.2f}) 以控制gamma")
                except:
                        print('please check')

                # 检查是否已经减少了足够的gamma
                if gamma_reduced <= gamma_reduction_needed:
                    print('reduce enough short gamma')
                    break

    # Drop the temporary gamma contribution column
    df = df.drop(columns=['gamma_dollar_contribution'])


    return df

def calculate_risk_exposures(df):
    """
    Calculate various risk exposures for options positions based on fitted volatility Greeks calculated using BSM model.

    Args:
        df: DataFrame containing option data with positions and Greeks calculated using fitted_vol
        volume_multiple: Contract multiplier (e.g., 100 for standard equity options)

    Returns:
        DataFrame with added risk exposure columns
    """
    # Make a copy to avoid modifying the original
    df_risk = df.copy()

    # Calculate delta exposure in dollar terms
    df_risk['delta_dollar'] = df_risk['volume_multiple'] * df_risk['final_underly_mean_mid'] * (
            df_risk['call_delta'] * df_risk['call_qty'] +
            df_risk['put_delta'] * df_risk['put_qty']
    )

    # Calculate gamma exposure in dollar terms (per 1% move)
    # Using the unified gamma from the updated calculation
    df_risk['gamma_dollar'] = df_risk['final_underly_mean_mid'] * df_risk['final_underly_mean_mid'] * \
                              df_risk['gamma'] * (df_risk['call_qty'] + df_risk['put_qty']) * \
                              df_risk['volume_multiple'] / 100

    # Calculate vega exposure in dollar terms (per 1% vol move)
    # Using the unified vega from the updated calculation
    df_risk['vega_dollar'] = df_risk['vega'] * (df_risk['call_qty'] + df_risk['put_qty']) * \
                             df_risk['volume_multiple']

    # Calculate theta exposure in dollar terms (per day)
    df_risk['theta_dollar'] = (df_risk['call_theta'] * df_risk['call_qty'] +
                               df_risk['put_theta'] * df_risk['put_qty']) * \
                              df_risk['volume_multiple']

    # Calculate rho exposure in dollar terms (per 1% rate change)
    df_risk['rho_dollar'] = (df_risk['call_rho'] * df_risk['call_qty'] +
                             df_risk['put_rho'] * df_risk['put_qty']) * \
                            df_risk['volume_multiple']

    return df_risk


def basic_vol_curve_trading_opportunities(df_latest_price,time_stamp,fresh_time_mins=0, vol_threshold=0.02,profit_margin=0, position_size=1,position_limit=50,bid_ask_spread=5,signal_exists_time=0):
    """
    Identifies and executes trading opportunities based on volatility curve analysis.

    This function evaluates the difference between fitted volatility and market volatility (bid/ask)
    to determine trading signals for options. It adjusts the order quantities for calls and puts
    based on the identified signals while considering constraints like bid-ask spread, data freshness,
    and position limits.

    Args:
        df_latest_price: DataFrame containing the latest option data, including strikes, volatilities, and Greeks.
        time_stamp: Current timestamp for the trading session.
        fresh_time_mins: Maximum age (in minutes) of data to consider as fresh for trading.
        vol_threshold: Minimum volatility difference required to trigger a trade.
        profit_margin: Minimum profit margin (volatility difference * vega) required to execute a trade.
        position_size: Number of contracts to trade when a signal is triggered.
        position_limit: Maximum allowable position size for each option.
        bid_ask_spread: Minimum bid-ask spread required to consider an option for trading.
        signal_exists_time: Minimum duration (in time steps) a signal must persist before executing a trade.

    Returns:
        Updated DataFrame with modified `call_current_order_qty` and `put_current_order_qty` values
        to reflect the trading decisions.

    """

    # Make a copy to avoid modifying the original
    df = df_latest_price.copy()

    # Process each strike price
    for idx, row in df.iterrows():
        strike = row['strike']
        underlying = row['final_underly_mean_mid']

        # Determine if the option is OTM, only trade OTM options
        is_call_otm = strike > underlying
        is_put_otm = strike < underlying

        # Check if the data for call and put options is fresh
        if_call_data_fresh=False
        if_put_data_fresh=False
        if not pd.isna(row['call_time']):
            if_call_data_fresh= time_stamp - row['call_time']<=pd.Timedelta(minutes=fresh_time_mins)
        if not pd.isna(row['put_time']):
            if_put_data_fresh= time_stamp - row['put_time']<=pd.Timedelta(minutes=fresh_time_mins)

        # Check if the bid-ask spread is within acceptable limits
        if_call_can_trade = (row['call_mean_ask'] - row['call_mean_bid']) >= bid_ask_spread
        if_put_can_trade = (row['put_mean_ask'] - row['put_mean_bid']) >= bid_ask_spread

        # Evaluate trading signals for call options
        if is_call_otm and if_call_data_fresh:  # Only trade OTM calls
            # Long signal: fitted volatility is significantly higher than ask volatility
            long_signal = (row['fitted_vol'] > row['call_ask_vol'] + vol_threshold)&((row['fitted_vol']-row['call_ask_vol'])*100*row['vega']>profit_margin)
            # Short signal: fitted volatility is significantly lower than bid volatility
            short_signal = (row['fitted_vol'] < row['call_bid_vol'] - vol_threshold)&((row['call_bid_vol']-row['fitted_vol'])*100*row['vega']>profit_margin)

            # update how long does the trading opportunitu last
            if long_signal:
                if row['call_trade_signal']<0:
                    df.at[idx, 'call_trade_signal'] =0
                else:
                    df.at[idx, 'call_trade_signal'] += 1
            if short_signal:
                if row['call_trade_signal']>0:
                    df.at[idx, 'call_trade_signal'] =0
                else:
                    df.at[idx, 'call_trade_signal'] -= 1


            if long_signal and if_call_can_trade and row['call_trade_signal']>=signal_exists_time and row['call_qty']+position_size <= position_limit:
                # Go long on call
                df.at[idx, 'call_current_order_qty'] = position_size
                print(
                    f"OPEN LONG CALL at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")
            elif short_signal and if_call_can_trade and row['call_trade_signal']<=-signal_exists_time and row['call_qty']-position_size >= -position_limit:
                # Go short on call
                df.at[idx, 'call_current_order_qty'] = -position_size
                print(
                    f"OPEN SHORT CALL at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")

        # Update put position
        if is_put_otm and if_put_data_fresh:  # Only trade OTM puts

            long_signal = (row['fitted_vol'] > row['put_ask_vol'] + vol_threshold)&((row['fitted_vol']-row['put_ask_vol'])*100*row['vega']>profit_margin)
            short_signal = (row['fitted_vol'] < row['put_bid_vol'] - vol_threshold)&((row['put_bid_vol']-row['fitted_vol'])*100*row['vega']>profit_margin)

            if long_signal:
                if row['put_trade_signal']<0:
                    df.at[idx, 'put_trade_signal'] =0
                else:
                    df.at[idx, 'put_trade_signal'] += 1
            if short_signal:
                if row['put_trade_signal']>0:
                    df.at[idx, 'put_trade_signal'] =0
                else:
                    df.at[idx, 'put_trade_signal'] -= 1


            if long_signal and if_put_can_trade and row['put_trade_signal']>=signal_exists_time and row['put_qty'] +position_size <= position_limit:
                # Go long on put
                df.at[idx, 'put_current_order_qty'] = position_size
                # df.at[idx, 'put_trade_qty'] = position_size
                # df.at[idx, 'put_qty'] += position_size
                print(
                    f"OPEN LONG PUT at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")
            elif short_signal and if_put_can_trade and row['put_trade_signal']<=-signal_exists_time and row['put_qty'] - position_size >= -position_limit:
                # Go short on put
                df.at[idx, 'put_current_order_qty'] = -position_size
                # df.at[idx, 'put_trade_qty'] = -position_size
                # df.at[idx, 'put_qty'] -= position_size
                print(
                    f"OPEN SHORT PUT at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")

        # check if current position fulfilled the condition to get out/ take profits
        if row['call_qty'] > 0 and if_call_data_fresh and  if_call_can_trade and row['fitted_vol'] <= row['final_vol']:
            # Close all position
            print(
                f"CLOSE Long CALL at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")
            df.at[idx, 'call_current_order_qty'] = -row['call_qty']

        elif row['call_qty'] < 0 and if_call_data_fresh and if_call_can_trade and row['fitted_vol'] >= row['final_vol']:
            # Close all position
            print(
                f"CLOSE short CALL at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")

            df.at[idx, 'call_current_order_qty'] = -row['call_qty']


        if row['put_qty'] > 0 and if_put_data_fresh and if_put_can_trade and row['fitted_vol'] <= row['final_vol']:
            # Close position
            print(
                f"CLOSE Long PUT at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")
            df.at[idx, 'put_current_order_qty'] = -row['put_qty']


        elif row['put_qty'] < 0 and if_put_data_fresh and if_put_can_trade and row['fitted_vol'] >= row['final_vol']:
            # Close position
            print(
                f"CLOSE short PUT at strike {strike}, fitted_vol: {row['fitted_vol']:.4f}, final_vol: {row['final_vol']:.4f}")
            df.at[idx, 'put_current_order_qty'] = -row['put_qty']


    return df


def simple_match_order(current_df,fullfilled_ratio=1.0):
    #TODO: can add order price and check if order price could be fullfilled or add some randomness

    df = current_df.copy()
    # simple version assume fullfilled based on fix fullfilled_ratio
    df['call_qty']=df['call_qty']+(df['call_previous_order_qty']*fullfilled_ratio).astype(int)
    df['put_qty']=df['put_qty']+(df['put_previous_order_qty']*fullfilled_ratio).astype(int)

    return df


def calculate_pnl(current_df, previous_df=None,option_fee=1,order_type='taker'):
    """
    Calculate P&L and some basic attribution based on volatility changes.

    Args:
        current_df: Current DataFrame with option data
        previous_df: Previous DataFrame for comparison (if None, P&L is 0)
        option_fee each trade single side
        order_type: Order type taker, maker,mid, twap, free
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
    total_bid_ask_fee=0
    total_option_fee=0
    total_delta_pnl=0.0
    total_gamma_pnl=0.0
    total_vega_pnl=0.0
    total_theta_pnl=0.0

    # If there's no previous data, return zeros
    if previous_df is None:
        return df, total_pnl,total_call_pnl,total_put_pnl,total_bid_ask_fee,total_option_fee,total_delta_pnl,total_gamma_pnl,total_vega_pnl,total_theta_pnl

    # Ensure both dataframes have the same strikes
    common_strikes = set(df['strike']).intersection(set(previous_df['strike']))

    for strike in common_strikes:
        curr_row = df[df['strike'] == strike].iloc[0]
        prev_row = previous_df[previous_df['strike'] == strike].iloc[0]

        curr_underlying_price = curr_row['final_underly_mean_mid']
        prev_underlying_price = prev_row['final_underly_mean_mid']
        underlying_price_pct_change = (curr_underlying_price - prev_underlying_price) / prev_underlying_price

        curr_vol = curr_row['fitted_vol']
        prev_vol = prev_row['fitted_vol']
        vol_change = curr_vol - prev_vol
        time_change = curr_row['t']-prev_row['t']

        idx = df[df['strike'] == strike].index[0]

        #上一分钟有持仓就结算收益
        # Calculate call P&L based on vega and vol change
        if prev_row['call_qty'] != 0:
            call_price_change = curr_row['call_mean_mid'] - prev_row['call_mean_mid']
            # P&L = quantity * vega * vol change * 100 (as vega is per 1% change)
            call_pnl = prev_row['call_qty'] * call_price_change * prev_row['volume_multiple']
            df.at[idx, 'call_pnl'] = call_pnl
            total_pnl += call_pnl
            total_call_pnl+=call_pnl
            delta_pnl_call = prev_row['call_delta'] * prev_row['call_qty'] * prev_underlying_price * underlying_price_pct_change * prev_row[
                'volume_multiple']
            # Gamma P&L (approximation for second-order price change)
            gamma_pnl_call =0.5 * prev_row['gamma'] * (prev_row['call_qty']) * (prev_underlying_price ** 2) * prev_row[
                'volume_multiple'] * (underlying_price_pct_change ** 2)
            # Vega P&L
            vega_pnl_call = prev_row['vega'] * prev_row['call_qty'] * vol_change * 100 * prev_row['volume_multiple']
            # Theta P&L
            theta_pnl_call = prev_row['call_theta'] * prev_row['call_qty'] * time_change * prev_row['volume_multiple']*365

            total_delta_pnl += delta_pnl_call
            total_gamma_pnl += gamma_pnl_call
            total_vega_pnl += vega_pnl_call
            total_theta_pnl += theta_pnl_call


        # Calculate put P&L based on vega and vol change
        if prev_row['put_qty'] != 0:
            put_price_change = curr_row['put_mean_mid'] - prev_row['put_mean_mid']
            # P&L = quantity * vega * vol change * 100 (as vega is per 1% change)
            put_pnl = prev_row['put_qty'] * put_price_change * prev_row['volume_multiple']
            df.at[idx, 'put_pnl'] = put_pnl
            total_pnl += put_pnl
            total_put_pnl += put_pnl

            # Delta P&L
            delta_pnl_put = prev_row['put_delta'] * prev_row['put_qty'] * prev_underlying_price * underlying_price_pct_change * prev_row[
                'volume_multiple']

            # Gamma P&L (approximation for second-order price change)
            gamma_pnl_put = 0.5 * prev_row['gamma'] * prev_row['put_qty'] * (prev_underlying_price ** 2) * (
                        underlying_price_pct_change ** 2) * prev_row['volume_multiple']

            # Vega P&L
            vega_pnl_put = prev_row['vega'] * prev_row['put_qty'] * vol_change * 100 * prev_row['volume_multiple']

            # Theta P&L
            theta_pnl_put = prev_row['put_theta'] * prev_row['put_qty'] * time_change * prev_row['volume_multiple']*365

            total_delta_pnl += delta_pnl_put
            total_gamma_pnl += gamma_pnl_put
            total_vega_pnl += vega_pnl_put
            total_theta_pnl += theta_pnl_put

        #上一分钟有下单就结算交易费用
        if curr_row['call_previous_order_qty'] != 0:
            if order_type=='taker':
                if curr_row['call_previous_order_qty']>0:
                    call_bid_ask_fee= -abs(curr_row['call_previous_order_qty']) * (curr_row['call_mean_ask'] - curr_row['call_mean_mid'])* curr_row['volume_multiple']
                else:
                    call_bid_ask_fee = -abs(curr_row['call_previous_order_qty']) * (curr_row['call_mean_mid']-curr_row['call_mean_bid'] )* curr_row['volume_multiple']
            if order_type=='maker':
                if curr_row['call_previous_order_qty']>0:
                    call_bid_ask_fee= -abs(curr_row['call_previous_order_qty']) * (curr_row['call_mean_bid'] - curr_row['call_mean_mid'])* curr_row['volume_multiple']
                else:
                    call_bid_ask_fee = -abs(curr_row['call_previous_order_qty']) * (curr_row['call_mean_mid']-curr_row['call_mean_ask'] )* curr_row['volume_multiple']

            elif order_type=='mid':
                call_bid_ask_fee= 0

            elif order_type=='twap':
                call_bid_ask_fee = -abs(curr_row['call_previous_order_qty']) * (curr_row['call_twap'] - curr_row['call_mean_mid']) * curr_row['volume_multiple']
            else:
                #上一时刻的mid price成交
                call_bid_ask_fee = -abs(curr_row['call_previous_order_qty']) * (prev_row['call_mean_mid']-curr_row['call_mean_bid'] )* curr_row['volume_multiple']

            if np.isnan(call_bid_ask_fee):
                print('check nan bid ask fee')

            call_option_fee=-option_fee*abs(curr_row['call_previous_order_qty'])
            df.at[idx, 'call_bid_ask_fee'] = call_bid_ask_fee
            df.at[idx, 'call_option_fee'] = call_option_fee
            total_bid_ask_fee += call_bid_ask_fee
            total_option_fee+=call_option_fee

        if curr_row['put_previous_order_qty'] != 0:
            if order_type=='taker':
                if curr_row['put_previous_order_qty']>0:
                    put_bid_ask_fee= -abs(curr_row['put_previous_order_qty']) * (curr_row['put_mean_ask'] - curr_row['put_mean_mid'])* curr_row['volume_multiple']
                else:
                    put_bid_ask_fee = -abs(curr_row['put_previous_order_qty']) * (curr_row['put_mean_mid']-curr_row['put_mean_bid'] )* curr_row['volume_multiple']
            if order_type=='maker':
                if curr_row['put_previous_order_qty']>0:
                    put_bid_ask_fee= -abs(curr_row['put_previous_order_qty']) * (curr_row['put_mean_bid'] - curr_row['put_mean_mid'])* curr_row['volume_multiple']
                else:
                    put_bid_ask_fee = -abs(curr_row['put_previous_order_qty']) * (curr_row['put_mean_mid']-curr_row['put_mean_ask'] )* curr_row['volume_multiple']
            elif order_type=='mid':
                put_bid_ask_fee= 0
            elif order_type=='twap':
                put_bid_ask_fee = -abs(curr_row['put_previous_order_qty']) * (curr_row['put_twap'] - curr_row['put_mean_mid']) * curr_row['volume_multiple']
            else:
                # 上一时刻的mid price成交
                put_bid_ask_fee = -abs(curr_row['put_previous_order_qty']) * (prev_row['put_mean_mid']-curr_row['put_mean_bid'] )* curr_row['volume_multiple']

            if np.isnan(put_bid_ask_fee):
                print('check nan bid ask fee')
            put_option_fee = -option_fee * abs(prev_row['put_previous_order_qty'])
            df.at[idx, 'put_bid_ask_fee'] = put_bid_ask_fee
            df.at[idx, 'put_option_fee'] = put_option_fee
            total_bid_ask_fee += put_bid_ask_fee
            total_option_fee += put_option_fee

        # Calculate total P&L for this strike
        df.at[idx, 'total_pnl'] = df.at[idx, 'call_pnl'] + df.at[idx, 'put_pnl']

    return df, total_pnl,total_call_pnl,total_put_pnl,total_bid_ask_fee,total_option_fee,total_delta_pnl,total_gamma_pnl,total_vega_pnl,total_theta_pnl


def plot_volatility_curves(df, time_stamp, save_dir=None,if_show=False):
    """
    Plot volatility curves including fitted vol, call/put vols (mid, bid, ask)

    Args:
        df: DataFrame containing volatility data
        time_stamp: Timestamp of the data (used in title and filename)
        save_dir: Directory to save the plot (optional)
        if_show: Boolean to show the plot or not
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


def vol_fitting(df_cut, time_stamp,fresh_time_mins=0):

    """
    Fits a volatility curve using a subset of option data.

    This function filters the input DataFrame for fresh data, calculates vega for each option,
    and fits a piecewise linear model to the volatility curve based on the strike prices.

    Args:
        df_cut: DataFrame containing option data, including strikes, volatilities, and Greeks.
        time_stamp: Current timestamp for the trading session.
        fresh_time_mins: Maximum age (in minutes) of data to consider as fresh for fitting.

    Returns:
        The original DataFrame with fitted volatility parameters.
    """

    global time_stamp_list, ATM_vol, Skew_list, Kp_list, SKP_list, Kc_list, SKC_list
    global Fwd, t

    df_cut = df_cut.reset_index(drop=True)
    df_original = deepcopy(df_cut)
    x_original = np.array(df_original['strike'])
    r = default_r
    t = df_cut['t'].values[0]
    Fwd = df_cut['final_underly_mean_mid'].values[0]
    # Initialize lists to store filtered strike prices and volatilities
    x_list = []
    y_list = []
    # Filter the DataFrame to include only rows with non-NaN final volatilities
    df_cut = df_cut.dropna(subset=['final_vol'])
    # Further filter the DataFrame to include only fresh data based on the provided time threshold
    df_cut = df_cut[(time_stamp-df_cut['final_vol_time'])<=pd.Timedelta(minutes=fresh_time_mins)]
    # Calculate d1 for each option using the Black-Scholes formula
    df_cut = df_cut.reset_index(drop=True)
    d1 = (np.log(df_cut["final_underly_mean_mid"].astype(float) / df_cut['strike'].astype(float)) + (r + df_cut['final_vol'] ** 2 / 2) *
              df_cut['t']) / (df_cut['final_vol'] * np.sqrt(df_cut["t"].astype(float)))
    d1=d1.astype(float)
    # Calculate vega for each option and normalize it by the maximum vega
    df_cut['vega'] = df_cut["final_underly_mean_mid"].astype(float) * np.sqrt(df_cut["t"].astype(float)) * norm.pdf(d1)
    max_vega = df_cut['vega'].max()
    df_cut['vega'] = df_cut['vega']/max_vega

    # Iterate through the filtered DataFrame to prepare data for curve fitting
    for i in range(0, len(df_cut)):
        times = max(int(round(df_cut.loc[i]['vega']*100)), 10)
        x_list = x_list + [df_cut.loc[i]['strike']] * times
        y_list = y_list + [df_cut.loc[i]['final_vol']] * times
    x = np.array(x_list)
    y = np.array(y_list)

    ###########
    # # Fit a piecewise linear model to the volatility curve
    try:
        popt, pcov = curve_fit(piecewise_linear, x, y, maxfev=1000)
        # 拟合系数存下来研究
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

    return df_original


def calc_bs_fair_and_greeks(df_price: pd.DataFrame):

    r=default_r
    df=df_price.copy()
    S = df["final_underly_mean_mid"].values
    K = df["strike"].values
    t = df["t"].values
    vol = df["fitted_vol"].values  # using fitted vol
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    df[f"call_fair"] = (
        norm.cdf(d1) * S - norm.cdf(d2) * K * np.exp(-r * t)
    )
    df[f"put_fair"] = (
        norm.cdf(-d2) * K * np.exp(-r * t) - norm.cdf(-d1) * S
    )
    df[f"call_delta"] = norm.cdf(d1)
    df[f"put_delta"] = df[f"call_delta"] -1
    df[f"call_theta"] = -((S * norm.pdf(d1) * vol) / (2 * np.sqrt(t))) -(r * K * np.exp(-r * t) * norm.cdf(d2))/365
    df[f"put_theta"] = -((S * norm.pdf(d1) * vol) / (2 * np.sqrt(t))) + (r * K * np.exp(-r * t) * norm.cdf(-d2))/365
    df[f"call_rho"] = K * t * np.exp(-r * t) * norm.cdf(d2)/100
    df[f"put_rho"] = -K * t * np.exp(-r * t) * norm.cdf(-d2)/100
    df[f"gamma"] = norm.pdf(d1) / (S * vol * np.sqrt(t))
    df[f"vega"] = S * norm.pdf(d1) * np.sqrt(t) / 100

    return df


# Vol smile model
def piecewise_linear(x, ATM, Skew, Kp, SKP, Kc, SKC):
    log_term = np.log(Fwd / x) / np.sqrt(t)
    left = ATM + Skew * log_term + 0.5 * Kp * log_term**2 + (5 / 3) * SKP * log_term**3
    right = ATM + Skew * log_term + 0.5 * Kc * log_term**2 - (5 / 3) * SKC * log_term**3
    return np.where(x <= Fwd, left, right)



if __name__ == '__main__':

    global default_r

    default_r = 0.03  # risk-free rate

    # Define the underlying asset (e.g., CF, OI, RM, SR)
    # underlying = 'CF'
    # underlying = 'OI'
    # underlying = 'RM'
    underlying = 'SR'

    # Set the volume multiplier, same for options and underlying asset
    # check on Zhengzhou Commodity Exchange website
    volume_multiple=10
    if underlying == 'CF':
        volume_multiple=5

    # Option fee per trade
    option_fee = 1.5
    if underlying == 'RM':
        option_fee=0.8

    # to determine if the data is fresh and can be used to fit vol curve and trade
    # 0表示只有当前分钟更新了的期权才是有效的，才能下单，1表示上一分钟更新的也行
    # fresh_time_mins=0
    fresh_time_mins=1

    # maximum bid ask spread required to place order
    # max_bid_ask_spread=5
    max_bid_ask_spread=2
    # max_bid_ask_spread=1


    # the price that the order is fullfilled
    # 当前这一分钟的mid price成交
    # order_type='free'
    # 下一分钟的平均对手价
    # order_type='taker'
    # 下一分钟的平均挂单价
    # order_type='maker'
    # 下一分钟的平均中间价
    # order_type='mid'
    # 下一分钟的twap
    order_type='twap'

    # minimum profit margin required profit_margin = vol_spread * vega
    profit_margin=10
    # profit_margin=20
    # profit_margin=40
    # profit_margin=100

    # minimum vol spread required
    # vol_threshold=0.01
    vol_threshold=0.005

    # ratio of order been fullfilled, simplified in current version
    fullfilled_ratio=1.0
    # order qty each time for each options instruments
    option_order_qty=1
    # maximum position for each options instruments
    option_position_limit=10


    # risk exposure limit, currently just adjust order qty to avoid exceed gamma exposure, use underlying to hedge delta exposure
    delta_dollar_hedge_limit = 5*1e5
    # gamma_dollar_hedge_limit = 1*1e5
    gamma_dollar_hedge_limit = 5*1e4

    # Define the minimum duration a signal must persist before trading
    # signal_exists_time=0
    # signal_exists_time=1
    signal_exists_time=3
    # signal_exists_time=5

    # Load the option data from the Parquet file, this a the file with pre-washed implied vol using monte carlo
    file_name = 'opt_test_data/washed_vol/montecarlo_vol_{}_data.parquet'.format(underlying)
    # file_name = 'opt_test_data/monte_carlo_results_CF.parquet'.format(underlying)

    df = pd.read_parquet(file_name, engine='pyarrow')

    df = df.sort_values(by=['minute_str', 'strike']).reset_index(drop=True)

    mean_bid_ask_spread=np.mean(df['mean_ask']-df['mean_bid'])
    print('mean bid ask spread {}'.format(mean_bid_ask_spread))
    # print('pct that full filled bid ask spread:{}'.format(np.mean(df['mean_ask']-df['mean_bid']<=max_bid_ask_spread)))

    # df[:1000].to_csv('mc_sample_data.csv')

    strike_counts_per_minute = df.groupby(['minute_str', 'strike']).size().reset_index(name='count')
    if sum(strike_counts_per_minute['count']!=2)>0:
        print('call put存在无法配对的情况,占比为{}'.format(sum(strike_counts_per_minute['count']!=2)/len(strike_counts_per_minute)))

    maturity_diff_set=set(df['maturity'].diff())
    print(maturity_diff_set)
    #maturity为剩余自然日

    # use smaller sample to test
    #差不多20个交易日
    # df=df[:100000]
    # df=df[:30000]

    df['minute_str'] = pd.to_datetime(df['minute_str'])
    df = df.sort_values(by='minute_str')

    num_minutes=len(df['minute_str'].unique())

    # 快速处理nan value
    # Fill mid_vol
    df['mid_vol'] = np.where(
        df['mid_vol'].isna(),
        np.where(
            df['bid_vol'].notna() & df['ask_vol'].notna(),
            (df['bid_vol'] + df['ask_vol']) / 2,
            np.where(df['bid_vol'].notna(), df['bid_vol'], df['ask_vol'])
        ),
        df['mid_vol']
    )

    # Fill bid_vol
    df['bid_vol'] = np.where(
        df['bid_vol'].isna() & df['mid_vol'].notna() & df['ask_vol'].notna(),
        2 * df['mid_vol'] - df['ask_vol'],
        df['bid_vol']
    )

    # Fill ask_vol
    df['ask_vol'] = np.where(
        df['ask_vol'].isna() & df['mid_vol'].notna() & df['bid_vol'].notna(),
        2 * df['mid_vol'] - df['bid_vol'],
        df['ask_vol']
    )

    # Columns we want to track for each call/put leg
    base_columns = ['mean_ask', 'mean_bid', 'cumcashvol', 'mean_mid', 'cumvolume', 'openinterest', 'twap','bid_vol','ask_vol','mid_vol']

    # Initialize empty DataFrame for tracking the latest option data
    extra_columns = ['maturity','t', 'final_underly_mean_mid']
    df_latest_price = pd.DataFrame(columns=['strike'] +
                                           [f'call_{col}' for col in base_columns] + ['call_time'] +
                                           [f'put_{col}' for col in base_columns] + ['put_time'] +
                                           extra_columns)
    df_latest_price['volume_multiple'] = volume_multiple
    df_latest_price['call_qty']=0
    df_latest_price['put_qty']=0
    # df_latest_price['call_trade_qty']=0
    # df_latest_price['put_trade_qty']=0

    df_latest_price['call_current_order_qty'] = 0
    df_latest_price['put_current_order_qty'] = 0
    df_latest_price['call_previous_order_qty'] = 0
    df_latest_price['put_previous_order_qty'] = 0
    # 用于比较交易信号强度和交易优先级
    df_latest_price['call_trade_signal'] = 0
    df_latest_price['put_trade_signal'] = 0

    df_latest_price['call_bid_ask_fee']=0
    df_latest_price['put_bid_ask_fee']=0
    df_latest_price['call_option_fee']=0
    df_latest_price['put_option_fee']=0
    df_latest_price['call_pnl']=0
    df_latest_price['put_pnl']=0

    greek_suffixes = ['_delta', '_theta', '_rho', '_fair']
    for col in greek_suffixes:
        df_latest_price[f"call{col}"] = np.nan
        df_latest_price[f"put{col}"] = np.nan
    df_latest_price[f"gamma"] = np.nan
    df_latest_price[f"vega"] = np.nan

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
    df_all_minute_price=pd.DataFrame()
    # store them in a dict
    df_history = {}
    df_minute_pnl_and_risk=pd.DataFrame()

    # Initialize variables for underlying hedging, currently not considering the fee using underlying to hedge
    underlying_hedge_qty=0
    df['underlying_instr']=df['Instrument'].apply(lambda x:x[:6])
    current_instr = df['underlying_instr'].iloc[0]
    previous_instr = df['underlying_instr'].iloc[0]
    if_new_undelrying_instr=False

    #用于检查信号持续时间
    max_call_long_trade_signal = 0
    max_call_short_trade_signal = 0
    max_put_long_trade_signal = 0
    max_put_short_trade_signal = 0
    for minute, group in tqdm(df.groupby('minute_str'), total=num_minutes,
                              desc="Processing minutes", unit="minute"):
        print(f"Processing minute: {minute}")
        current_instr = group['underlying_instr'].iloc[0]
        if current_instr!=previous_instr:
            print('underlying instr change')
            #clear current price info
            df_latest_price=df_latest_price[:0]
            underlying_hedge_qty=0
            #TODO: clear position and calculate the cost
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
                empty_row['call_current_order_qty'] = 0
                empty_row['put_current_order_qty'] = 0
                empty_row['call_previous_order_qty'] = 0
                empty_row['put_previous_order_qty'] = 0
                empty_row['call_trade_signal'] = 0
                empty_row['put_trade_signal'] = 0
                empty_row['call_bid_ask_fee'] = 0
                empty_row['put_bid_ask_fee'] = 0
                empty_row['call_option_fee'] = 0
                empty_row['put_option_fee'] = 0
                empty_row['call_pnl'] = 0
                empty_row['put_pnl'] = 0
                empty_row['volume_multiple'] = volume_multiple
                df_latest_price = pd.concat([df_latest_price, empty_row.to_frame().T])
                df_latest_price = df_latest_price.sort_values(by=['strike']).reset_index(drop=True)

            for k, v in update_data.items():
                df_latest_price.loc[df_latest_price['strike'] == strike, k] = v
                # df_latest_price.at[strike, k] = v

        df_latest_price = df_latest_price.sort_values(by=['strike']).reset_index(drop=True)
        # df_latest_price['strike_price']=df_latest_price.index
        df_latest_price['final_vol'] = np.where(
            df_latest_price['strike'] >= df_latest_price['final_underly_mean_mid'],
            df_latest_price['call_mid_vol'],
            df_latest_price['put_mid_vol']
        )

        df_latest_price['final_vol_time'] = np.where(
            df_latest_price['strike'] >= df_latest_price['final_underly_mean_mid'],
            df_latest_price['call_time'],  # call mid vol placeholder
            df_latest_price['put_time']  # put mid vol placeholder
        )

        # fit the vol curve
        df_latest_price=vol_fitting(df_latest_price,minute,fresh_time_mins)

        #TODO: upgrade greeks calculation method fo american options
        df_latest_price=calc_bs_fair_and_greeks(df_latest_price)
        underlying_price=df_latest_price['final_underly_mean_mid'].iloc[0]

        # check vol curve and see if any pattern exists
        # plot_volatility_curves(df_latest_price,minute,if_show=True)

        # initiate value
        minute_option_pnl=0
        call_pnl=0
        put_pnl=0
        previous_bid_ask_fee=0
        previous_option_fee=0
        delta_pnl=0
        gamma_pnl=0
        vega_pnl=0
        theta_pnl=0

        # match order, update qty and calculate the pnl
        # If we have previous data of current maturity options, match order, calculate P&L and do basic attribution
        if (len(df_history) > 1) and (not if_new_undelrying_instr):
            prev_timestamp = list(df_history.keys())[-2]
            #TODO: upgrade order matching algorithm
            df_latest_price=simple_match_order(df_latest_price,fullfilled_ratio)
            df_latest_price, minute_option_pnl,call_pnl,put_pnl,previous_bid_ask_fee,previous_option_fee,delta_pnl,gamma_pnl,vega_pnl,theta_pnl = calculate_pnl(df_latest_price, df_history[prev_timestamp],option_fee=option_fee,order_type=order_type)

        #calculate current risk exposure
        df_latest_price = calculate_risk_exposures(df_latest_price)
        total_option_delta = df_latest_price['delta_dollar'].sum()
        total_gamma_dollar = df_latest_price['gamma_dollar'].sum()
        total_vega = df_latest_price['vega_dollar'].sum()
        total_theta = df_latest_price['theta_dollar'].sum()



        # Find trading opportunities at this moment and update quantities to trade in next minutes in previous matching order part

        df_latest_price = basic_vol_curve_trading_opportunities(df_latest_price,minute,fresh_time_mins, vol_threshold=vol_threshold,profit_margin=profit_margin,
                                                      position_size=option_order_qty,position_limit=option_position_limit,bid_ask_spread=max_bid_ask_spread,signal_exists_time=signal_exists_time)

        max_call_long_trade_signal= max(max_call_long_trade_signal,df_latest_price['call_trade_signal'][df_latest_price['call_trade_signal'] >= 0].max())
        max_call_short_trade_signal=min(max_call_short_trade_signal,df_latest_price['call_trade_signal'][df_latest_price['call_trade_signal'] <= 0].min())
        max_put_long_trade_signal=max(max_put_long_trade_signal,df_latest_price['put_trade_signal'][df_latest_price['put_trade_signal'] >= 0].max())
        max_put_short_trade_signal=min(max_put_short_trade_signal,df_latest_price['put_trade_signal'][df_latest_price['put_trade_signal'] <= 0].min())
        print(
            f"Max Call Long: {max_call_long_trade_signal}, "
            f"Max Call Short: {max_call_short_trade_signal}, "
            f"Max Put Long: {max_put_long_trade_signal}, "
            f"Max Put Short: {max_put_short_trade_signal}"
        )


        # adjust the order quantities to control risk exposure
        df_latest_price=basic_order_risk_control(df_latest_price,total_option_delta,total_gamma_dollar,delta_dollar_hedge_limit,gamma_dollar_hedge_limit)
        #TODO: 持仓exposure大了主动去hedge而不是有交易机会才控制


        if total_option_delta==0:
            underlying_hedge_qty=0
        # delta risk control, check risk and do hedging, assume is free first in current stage
        if (abs(underlying_hedge_qty*underlying_price+total_option_delta)>delta_dollar_hedge_limit):
            print('delta exceed limit, use underlying control delta exposure')
            underlying_hedge_qty= int(total_option_delta/(underlying_price*volume_multiple))


        df_latest_price['call_previous_order_qty']=df_latest_price['call_current_order_qty']
        df_latest_price['put_previous_order_qty']=df_latest_price['put_current_order_qty']
        df_latest_price['call_current_order_qty'] = 0
        df_latest_price['put_current_order_qty'] = 0

        # Store the current state into history dict
        df_history[minute] = df_latest_price.copy()

        print(f"Minute: {minute},underlying_price: {underlying_price} P&L: {minute_option_pnl:.2f} Risk Exposures at {minute}:Delta: ${total_option_delta:.2f} Gamma: ${total_gamma_dollar:.2f} Vega: ${total_vega:.2f} Theta: ${total_theta:.2f}")

        # store the data
        df_current_pnl_and_risk=pd.DataFrame({'time':[minute],'underlying_price':[underlying_price],'underlying_hedge_qty':[underlying_hedge_qty],
                                              'option_pnl':[minute_option_pnl],'call_pnl':[call_pnl], 'put_pnl':[put_pnl],'bid_ask_fee':[previous_bid_ask_fee],'option_fee':[previous_option_fee],
                                              'delta_pnl':[delta_pnl], 'gamma_pnl':[gamma_pnl], 'vega_pnl':[vega_pnl], 'theta_pnl':[theta_pnl],
        'total_option_delta':[total_option_delta], 'total_gamma_dollar':[total_gamma_dollar], 'total_vega':[total_vega],'total_theta':[total_theta]
        })
        df_minute_pnl_and_risk=pd.concat([df_minute_pnl_and_risk,df_current_pnl_and_risk])

        previous_instr=current_instr
        if_new_undelrying_instr = False

    # calculate the cummulated pnl
    df_minute_pnl_and_risk['call_cum_pnl']=df_minute_pnl_and_risk['call_pnl'].cumsum()
    df_minute_pnl_and_risk['put_cum_pnl']=df_minute_pnl_and_risk['put_pnl'].cumsum()
    df_minute_pnl_and_risk['option_cum_pnl']=df_minute_pnl_and_risk['option_pnl'].cumsum()
    df_minute_pnl_and_risk['bid_ask_cum_fee']= df_minute_pnl_and_risk['bid_ask_fee'].cumsum()
    df_minute_pnl_and_risk['option_cum_fee']= df_minute_pnl_and_risk['option_fee'].cumsum()
    df_minute_pnl_and_risk['underlying_hedge_pnl'] = df_minute_pnl_and_risk['underlying_hedge_qty'].shift(1)*df_minute_pnl_and_risk['underlying_price'].diff()*volume_multiple
    df_minute_pnl_and_risk['underlying_hedge_pnl'].iloc[0]=0
    df_minute_pnl_and_risk['underlying_hedge_cum_pnl']= df_minute_pnl_and_risk['underlying_hedge_pnl'].cumsum()
    df_minute_pnl_and_risk['total_cum_pnl_with_fee']=df_minute_pnl_and_risk['option_cum_pnl'] + df_minute_pnl_and_risk['bid_ask_cum_fee']
    df_minute_pnl_and_risk['total_cum_pnl_with_hedge']=df_minute_pnl_and_risk['option_cum_pnl'] + df_minute_pnl_and_risk['underlying_hedge_cum_pnl']
    df_minute_pnl_and_risk['total_cum_pnl_with_fee_and_hedge']=df_minute_pnl_and_risk['option_cum_pnl'] + df_minute_pnl_and_risk['bid_ask_cum_fee'] + df_minute_pnl_and_risk['underlying_hedge_cum_pnl']
    df_minute_pnl_and_risk['delta_cum_pnl']= df_minute_pnl_and_risk['delta_pnl'].cumsum()
    df_minute_pnl_and_risk['gamma_cum_pnl']= df_minute_pnl_and_risk['gamma_pnl'].cumsum()
    df_minute_pnl_and_risk['vega_cum_pnl']= df_minute_pnl_and_risk['vega_pnl'].cumsum()
    df_minute_pnl_and_risk['theta_cum_pnl']= df_minute_pnl_and_risk['theta_pnl'].cumsum()

    os.makedirs('backtest_results', exist_ok=True)
    df_minute_pnl_and_risk.to_csv('backtest_results/{}_pnl_and_risk_ftm{}_vol{}_pm{}_bas{}_set{}_ooq{}_opl{}_ot{}_fr{}_dhl{}_ghl{}.csv'.format(underlying,fresh_time_mins,vol_threshold,profit_margin,max_bid_ask_spread,signal_exists_time,option_order_qty,option_position_limit,order_type,fullfilled_ratio,delta_dollar_hedge_limit,gamma_dollar_hedge_limit),index=False)

    os.makedirs('backtest_results_plots', exist_ok=True)
    saved_name='backtest_results_plots/{}_ftm{}_vol{}_pm{}_bas{}_set{}_ooq{}_opl{}_ot{}_fr{}_dhl{}_ghl{}.html'.format(underlying,fresh_time_mins,vol_threshold,profit_margin,max_bid_ask_spread,signal_exists_time,option_order_qty,option_position_limit,order_type,fullfilled_ratio,delta_dollar_hedge_limit,gamma_dollar_hedge_limit)
    # make the plot
    plot_pnl_and_risk(
        df_minute_pnl_and_risk,
        save_html_path=saved_name,
        title='{} PnL and Risk Plot ftm{}_vol{}_pm{}_bas{}_set{}_ooq{}_opl{}_ot{}_fr{}_dhl{}_ghl{}'.format(underlying,fresh_time_mins,vol_threshold,profit_margin,max_bid_ask_spread,signal_exists_time,option_order_qty,option_position_limit,order_type,fullfilled_ratio,delta_dollar_hedge_limit,gamma_dollar_hedge_limit)
    )

    print()