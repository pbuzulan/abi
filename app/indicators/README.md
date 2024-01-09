# Technical Indicators Documentation

## Simple Moving Average (SMA)

**Method Name:** `calculate_sma()`
- SMA calculates the average of a selected range of prices, usually closing prices, by the number of periods in that range.
- Often used to identify trend direction.

## Exponential Moving Average (EMA)

**Method Name:** `calculate_ema()`
- EMA gives more weight to recent prices, reacting more quickly to price changes than the SMA.
- Useful for identifying trends in a more timely manner.

## Moving Average Convergence Divergence (MACD)

**Method Name:** `calculate_macd()`
- MACD is a trend-following momentum indicator.
- It shows the relationship between two moving averages of a security’s price.
- Can signal bullish or bearish trends.

## Relative Strength Index (RSI)

**Method Name:** `calculate_rsi()`
- RSI is a momentum oscillator that measures the speed and change of price movements.
- Ranges from 0 to 100, with high readings indicating overbought conditions and low readings indicating oversold conditions.

## Bollinger Bands

**Method Name:** `calculate_bollinger_bands()`
- Bollinger Bands are used to measure market volatility.
- Consist of a middle SMA and upper and lower bands that adjust based on volatility.
- Can signal potential overbought/oversold conditions.

## Stochastic Oscillator

**Method Name:** `calculate_stochastic()`
- Compares a security's closing price to its price range over a specific time period.
- Used to generate overbought and oversold trading signals.

## Average True Range (ATR)

**Method Name:** `calculate_atr()`
- Measures market volatility by decomposing the entire range of an asset price for that period.
- Helps to understand the enthusiasm or lack thereof among traders.

## Chaikin Money Flow (CMF)

**Method Name:** `calculate_cmf()`
- CMF combines price and volume to measure buying and selling pressure.
- A positive CMF indicates bullish pressure, while a negative CMF indicates bearish pressure.

## On-Balance Volume (OBV)

**Method Name:** `calculate_obv()`
- Uses volume flow to predict changes in stock price.
- The theory is that volume precedes price movement, so if a security is seeing an increasing OBV, it is generally thought to be bullish.

## Ichimoku Cloud

**Method Name:** `calculate_ichimoku()`
- Combines multiple indicators to create a "cloud" formation on the chart.
- Useful for identifying support and resistance levels, trend direction, and momentum.
- Includes components like Tenkan-sen, Kijun-sen, Senkou Span A and B, and Chikou Span.

## Williams %R

**Method Name:** `calculate_williams_r()`
- Measures the level of the close relative to the high-low range over a given period of time.
- Used to identify overbought or oversold conditions in a market.

## Commodity Channel Index (CCI)

**Method Name:** `calculate_cci()`
- CCI is a versatile indicator that can be used to identify a new trend or warn of extreme conditions.
- Typically used to identify cyclical trends in commodities, but can be applied to other asset classes.

## Momentum Indicator

**Method Name:** `calculate_momentum()`
- Measures the rate of change or speed of price movement of a security.
- Helps in understanding the strength of price movements, up or down.

## True Strength Index (TSI)

**Method Name:** `calculate_tsi()`
- A momentum oscillator that measures the direction and magnitude of price movements.
- Primarily used to identify overbought and oversold conditions.

## Money Flow Index (MFI)

**Method Name:** `calculate_mfi()`
- Combines price and volume to measure the strength of price changes and potentially predict future price changes.
- Similar to RSI but incorporates volume.

## Ultimate Oscillator

**Method Name:** `calculate_ultimate_oscillator()`
- Combines short-term, medium-term, and long-term price action into one oscillator.
- Aims to capture momentum across different timeframes.

## Standard Deviation

**Method Name:** `calculate_std_dev()`
- Measures the amount of variation or dispersion in a set of values.
- A high standard deviation indicates that the values are spread out over a wider range.

## Average Directional Movement Index Rating (ADXR)

**Method Name:** `calculate_adxr()`
- An indicator used to quantify trend strength.
- ADXR averages the values of ADX over a set period.

## Aroon Indicator

**Method Name:** `calculate_aroon()`
- Designed to predict when trends are likely to change direction.
- Measures the time between highs and the time between lows over a time period.

## Rate of Change (ROC)

**Method Name:** `calculate_roc()`
- Measures the percentage change in price from one period to the next.
- The ROC indicator might be used to confirm price moves or detect divergences.

## Market Sentiment Oscillator

**Method Name:** `calculate_market_sentiment_oscillator()`
- A combination of price change and volume change.
- Aims to capture the market's sentiment by analyzing both price and volume changes.

## Linear Regression Slope

**Method Name:** `calculate_linear_regression_slope()`
- Represents the rate of change in price over time.
- Useful for identifying trends: a positive slope indicates an uptrend, while a negative slope indicates a downtrend.

## Triangular Moving Average (TRIMA)

**Method Name:** `calculate_trima()`
- A moving average that gives more weight to middle values and less weight to values at the ends.
- Often smoother than a simple moving average.

## Volatility-Adjusted Moving Average (VAMA)

**Method Name:** `calculate_vama()`
- Adjusts a moving average based on the volatility of the market.
- Provides a more dynamic moving average that adapts to changing market conditions.

## Volume-Weighted Moving Average (VWMA)

**Method Name:** `calculate_vwma()`
- Incorporates volume into the moving average calculation.
- Gives more weight to periods with higher volume.

## Price Momentum Oscillator (PMO)

**Method Name:** `calculate_pmo()`
- A rate-of-change oscillator that smooths price changes and converts them into a momentum oscillator.
- Helps identify overbought and oversold conditions.

## Elder-Ray Index

**Method Name:** `calculate_elder_ray()`
- Developed by Dr. Alexander Elder and measures buying and selling pressure in the market.
- Combines EMA and bullish and bearish power to determine market sentiment.

## Keltner Channels

**Method Name:** `calculate_keltner_channels()`
- Similar to Bollinger Bands, Keltner Channels use average true range for band calculation.
- Helps to identify trend direction and reversals.

## Detrended Price Oscillator (DPO)

**Method Name:** `calculate_dpo()`
- Attempts to remove trend from price to make it easier to identify cycles.
- Used to identify overbought/oversold levels and short-term cycles.

## Accumulation/Distribution Line (A/D Line)

**Method Name:** `calculate_ad_line()`
- A volume-based indicator designed to measure the cumulative flow of money into and out of a security.
- Can be used to confirm trends or warn of reversals.

## Vortex Indicator (VI)

**Method Name:** `calculate_vortex_indicator()`
- An indicator designed to identify the start of a new trend or the continuation of an existing trend within price data.

## Volume Oscillator

**Method Name:** `calculate_volume_oscillator()`
- Measures the difference between two volume-based moving averages.
- Can signal bullish conditions when the oscillator is positive and bearish conditions when negative.

## Balance of Power (BOP)

**Method Name:** `calculate_bop()`
- Measures the strength of buyers against sellers in the market.
- Can be used to confirm or warn of market reversals.

## Hull Moving Average (HMA)

**Method Name:** `calculate_hma()`
- An extremely fast and smooth moving average that almost eliminates lag and manages to improve smoothing simultaneously.

## ZigZag Indicator

**Method Name:** `calculate_zigzag()`
- Identifies swing highs and swing lows in price fluctuations.
- Useful for filtering out market noise and focusing on the more significant trends and reversals.

## Donchian Channels

**Method Name:** `calculate_donchian_channels()`
- Measures the high and low of a price over a set period.
- Can be used for breakout strategies and trend following.

## Gann High Low Activator (HLA)

**Method Name:** `calculate_hla()`
- An indicator used to determine the direction of a trend and to define potential support/resistance levels.

## Chande Momentum Oscillator (CMO)

**Method Name:** `calculate_cmo()`
- Measures momentum on both up and down days.
- Can signal overbought and oversold conditions and possible reversals.

## Fisher Transform

**Method Name:** `calculate_fisher_transform()`
- Converts prices into a Gaussian normal distribution.
- Aims to pinpoint price reversals.

## Ultimate Oscillator

**Method Name:** `calculate_ultimate_oscillator()`
- Combines short-term, medium-term, and long-term price action into one oscillator.
- Aims to capture momentum across different timeframes.

## Ehlers Fisher Transform

**Method Name:** `calculate_ehlers_fisher_transform()`
- A variation of the Fisher Transform. It provides clearer turning points and overbought/oversold signals.

## Elder's Force Index (EFI)
**Method Name:** `calculate_efi()`
- Measures the power behind a price movement using price and volume.
- A higher EFI indicates strong buying pressure, while a lower EFI suggests strong selling pressure.

## Market Facilitation Index (MFI)
**Method Name:** `calculate_mfi()`
- Analyzes and visualizes the efficiency of price movement by comparing the price change and volume.
- Helps in identifying whether the movement is getting stronger or weaker.

## Average True Range Percentage (ATR %)
**Method Name:** `calculate_atr_percentage()`
- Expresses the Average True Range (ATR) indicator as a percentage of the closing price.
- Useful for comparing volatility across different price levels.

## Heikin-Ashi Technique
**Method Name:** `calculate_heikin_ashi()`
- Averages open, close, high, and low prices to create a smoothed candlestick chart.
- Often used to filter out market noise and identify trend direction more clearly.

## Trend Intensity Index (TII)
**Method Name:** `calculate_tii()`
- Measures the strength of a trend by comparing the current closing price to past closing prices.
- High values indicate a strong trend, while low values indicate a weak trend.

## Kaufman’s Adaptive Moving Average (KAMA)
**Method Name:** `calculate_kama()`
- An adaptive moving average designed to account for market noise or volatility.
- Adjusts its sensitivity to price changes.

##  Trend Oscillator (WTO)
**Method Name:** `calculate_wto()`
- A cyclical indicator used to identify overbought and oversold conditions.
- Helps in spotting the start and end of trends.

## Adaptive RSI (ARSI)
**Method Name:** `calculate_arsi()`
- A variation of the traditional RSI which adapts its period length based on recent market volatility.
- Provides a more responsive indicator sensitive to changing market conditions.

## Gann High Low Activator (HLA)
**Method Name:** `calculate_hla()`
- Identifies trend direction and reversals based on Gann's price and time balance theory.
- Often used to set stop-loss orders or trigger entries.

## Donchian Channels
**Method Name:** `calculate_donchian_channels()`
- Identifies the highest high and the lowest low over a set number of periods.
- Often used in breakout trading strategies.

## Adaptive Moving Average (AMA)
**Method Name:** `calculate_ama()`
- Adjusts its sensitivity based on the efficiency ratio which measures the degree of trend or direction in the price.
- Helps in identifying the start of a trend early.

## Volume-Price Trend (VPT)
**Method Name:** `calculate_vpt()`
- A cumulative line that combines volume and price to show the strength of price trends and potential reversals.
- Helps in confirming the strength of a trend.

## Elder-Ray Index
**Method Name:** `calculate_elder_ray()`
- Combines EMA and bullish/bearish power, assessing buying and selling pressures.
- Useful for identifying potential buy and sell points.

## Keltner Channels
**Method Name:** `calculate_keltner_channels()`
- Uses average true range to set channel distance, offering a dynamic view of support and resistance levels.
- Can be used for breakout and trend-following strategies.

## Detrended Price Oscillator (DPO)
**Method Name:** `calculate_dpo()`
- Removes the trend from price, making it easier to identify cycles and overbought/oversold levels.
- Useful for identifying the underlying cycle of market movements.

## Accumulation/Distribution Line (A/D Line)
**Method Name:** `calculate_ad_line()`
- A volume-based indicator intended to show the cumulative flow of money into and out of a security.
- Can signal if a trend is likely to continue or reverse.

## Vortex Indicator (VI)
**Method Name:** `calculate_vortex_indicator()`
- Designed to identify the start of a new trend or the continuation of an existing trend.
- Compares the range of current high and low prices to previous highs and lows.

## Volume Oscillator
**Method Name:** `calculate_volume_oscillator()`
- Measures the difference between a fast and slow volume moving average.
- A positive value suggests bullish momentum, while a negative value suggests bearish momentum.

## Balance of Power (BOP)
**Method Name:** `calculate_bop()`
- Indicates whether buyers or sellers are in control of the market at a given time.
- Useful for confirming trends or spotting reversals.

## Hull Moving Average (HMA)
**Method Name:** `calculate_hma()`
- A fast and smooth moving average that helps reduce the lag effect and increases accuracy.
- Useful for identifying the current market trend.

## Wave Trend Oscillator (WTO)
**Method Name:** `calculate_wto()`
- A volatility-based oscillator similar to MACD.
- Useful for spotting overbought and oversold conditions.

## Adaptive RSI (ARSI)
**Method Name:** `calculate_arsi()`
- An enhanced version of RSI that adjusts its period dynamically based on recent market volatility.
- Provides more accurate readings of overbought and oversold conditions.

## Gann High Low Activator (HLA)
**Method Name:** `calculate_hla()`
- Identifies trend direction and reversals based on Gann's price and time balance theory.
- Often used to set stop-loss orders or trigger entries.
