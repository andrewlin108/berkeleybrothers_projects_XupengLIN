Option Project by Xupeng LIN

Environment
	•	Developed using PyCharm.

Idea and Background
	•	The time span of the data is about 2.5 months, which is too short to build reliable low-frequency trading strategies (such as delta or gamma timing strategies).
	•	The data is also limited; assuming no external data sources are allowed, I decided to give up on low-frequency strategies.

Additional challenges:
	•	The dataset is incomplete at the minute level; it does not contain the latest price for every option at every minute.
	•	Only one maturity is available at any given time.

Based on these limitations, I decided to:
	•	Maintain a price DataFrame, updating it every minute.
	•	Fit a volatility curve at each timestamp.
	•	Trade based on deviations from the fitted volatility curve:
	•	Short options when implied vol > fitted vol.
	•	Long options when implied vol < fitted vol.
	•	Focus on OTM (out-of-the-money) options due to higher liquidity.

Notes:
	•	The options are American-style without dividend.
	•	Initially, I used Black-Scholes-Merton (BSM) to quickly explore patterns, but soon realized BSM is not appropriate for American options.
	•	I compared different methods and chose to calculate implied volatility using Monte Carlo simulation, although computational resources and time were limited.
	•	After obtaining implied volatilities, I implemented a more realistic backtest.

⸻

Current Status
	•	The current results are not good.
	•	Some sample plots are included.
	•	More plots can be found at:
GitHub Repository

⸻

File Descriptions

vol_model_comparison.ipynb
	•	Compare the time and results of different methods for calculating implied volatility.
	•	Concluded that Monte Carlo simulation is the preferred method.

trade_bsm_model.py
	•	Simple idealized strategy using BSM implied volatilities.
	•	Steps:
	•	Fit a volatility curve at each minute.
	•	If fitted vol > current vol → Long the option.
	•	If fitted vol < current vol → Short the option.
	•	Only OTM options are traded.
	•	Assumes mid-price trading (unrealistic).
	•	Observations:
	•	BSM is unsuitable — puts often have higher implied vol than calls at the same strike, even across bid-ask.
	•	However, in idealized conditions, steady vega PnL could be earned.

quick_pre_wash_data_mc_implied_vol.py
	•	Calculate implied volatilities using Monte Carlo simulation.
	•	Trade-offs:
	•	Used smaller Monte Carlo parameters to shorten computation time.
	•	Volatility estimates may be noisy.

trade_mc_vol.py
	•	Realistic backtest using Monte Carlo-implied volatilities.
	•	Same trading logic (long/short based on vol deviations).
	•	Real-world considerations added:
	•	Check data freshness before analysis or trading.
	•	Order placement assumptions: different fill ratios and fill prices (taker, maker, mid, TWAP).
	•	Minimum vol spread and minimum profit margin (vol spread × vega) required to place an order.
	•	Limit quantity per order and maximum position size.
	•	Include option trading fees.
	•	Adjust orders based on gamma exposure (use underlying to control delta — fees not considered yet).
	•	Plot the fitted vol curve.
	•	Basic PnL attribution for performance analysis.
	•	Result: Still unsatisfactory and needs improvements.

⸻

Areas for Improvement
	•	Calculate more accurate implied volatilities (using larger Monte Carlo sample sizes).
	•	Incorporate price and volatility prediction (time-series modeling).
	•	Control more Greeks simultaneously during order placement.
	•	Actively hedge positions that exceed risk limits.
	•	Include costs for using the underlying to hedge delta exposure.
	•	Improve order fulfillment modeling.
	•	Use more precise risk exposure calculations.
	•	Enhance PnL attribution accuracy.
	•	Analyze PnL across different delta exposures to find potential patterns.

⸻

Future Extensions
	•	Implement implied spot arbitrage based on put-call parity under the current framework.

⸻

Not Useful (Currently)

split_data.py
	•	Script for splitting data based on the underlying instrument.
	•	Not directly used in this project at this stage.
