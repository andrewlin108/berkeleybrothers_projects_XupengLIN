This is the option project written by Xupeng LIN

environment:
i use pycharm to develop all these programs

Idea:

the time spread is about 2 and a half month, it’s unrealistic to find some delta or gamma timming strategy, backtest period is too short for low frequency trading streategy. And the data provided is also limited, assuming i can not use other data, i give up the idea to do this kind of strategy.

As the data is not complete for every minutes, it does not contain the latest minute price of all options, so i consider maintain a price dataframe, and handle the data every minutes and refresh the table. And there is only my maturity in one specific time, i can only fit a vol curve, and i think maybe just trade based on the fitted vol curve, if the vol is higher then the fitted one, i short it. vice versa. And it seems that i can earn the vega pnl steadily in ideal case. And i just trade otm options as it has higher liquidity

And commodity options is american options, but has no dividend. I tried to use bsm first to quickly find some pattern i can make use of, but also find out that using bsm is not a good choice, so i make a comparison of different method i choose to use monte carlo to calculate the implied vol, but the time and my computation power is limited, so i just use some small parameter to save time. After i got the implied vol, i write the backtest with more realistic assumption. 


The current result is not good, i just put some sample plots.
For more plots, you can find it in. https://github.com/andrewlin108/berkeleybrothers_projects_XupengLIN.git


These are some explanation of the python or ipynb file.

vol_model_comparison.ipynb

Compare the time and check the result using different method to find the implied vol and decided to calculate implied volatility using Monte Carlo simulation


trade_bsm_model

simple basic ideal strategy, using bsm to find out the implied vol of option, and fit a vol curve. Assuming we can trade on mid price of this current minute(unrealistic), if the fitted vol is higher than the current option, long the option. If fitted vol is lower, then short the option. Only trade the otm options. i can visualize what the current vol curve looks like in this python file. and using bsm is obviously not a good choice, the put implied vol is uncommonly higher than the call implied vol of same strike even cross bid ask.

Result shows that vega pnl can be earned steadily, so we move to using monte carlo simulation.



quick_pre_wash_data_mc_implied_vol.py

Calculate implied volatility using Monte Carlo simulation
As my computational power and time is limited, i use small parameter to shorten the calculation time, but the vol calculation result may be volatile.


trade_mc_vol

trade based on implied volatility using Monte Carlo simulation. Same trading logic as before,  if the fitted vol is higher than the current option, long the option. If fitted vol is lower, then short the option.
But get into a more realistic scenario,

we consider if the data is fresh enough for us to use to analyse or trade
we assume we can only put order at this moment, and consider different fixed ratio of completing the order, and different price we filled the order(taker, maker, mid ,twap)
i take in to account the minimum vol spread, minimum profit margin(vol spread*vega) required to place order
i take into account how qty each order and maximum qty each option
i take into account the trading fee for option
i consider adjust the order based on the gamma exposure, we use underlying to control delta but not consider fee in this current stage and consider the volume multiple
i can check the plot of fitted vol curve using this code
i do some basic pnl attribution to help analyze

Currently the result is bad.

Place to improve:
calculate more accurate implied vol based on monte carlo(takes more time)
incorporate price and volatility prediction(in time series)
control more greeks at the same time when  placing order
actively hedge the risk if current position exceed risk limit
take into account the fee of using underlying to hedge
improve order fullfilled algorithm
use more accurate way to calculate risk exposure
pnl attribution is not accurate enough, can be improved
check the pnl from different delta, check if there is some pattern about winning or losing money

Can also do some other trading strategies such as implied spot arbitrage based on put call parity in my current framework too.

not useful:

split_data.py

split data based on its underlying instument

