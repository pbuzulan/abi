import pandas as pd
import numpy as np
import logging
import talib
import inspect

logger = logging.getLogger(__name__)


def calculate_sma(df, timeperiod):
    """
    Simple Moving Average (SMA)
    SMA is a technical indicator that averages the closing prices over a specified period.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df[f'SMA_{timeperiod}'] = talib.SMA(df['close'], timeperiod=timeperiod)
    return df


def calculate_ema(df, timeperiod):
    """
    Exponential Moving Average (EMA)
    EMA is a type of moving average that places a greater weight and significance on the most recent data points.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df[f'EMA_{timeperiod}'] = talib.EMA(df['close'], timeperiod=timeperiod)
    return df


def calculate_stochastic_oscillator(df, fastk_period, slowk_period, slowd_period):
    """
    Stochastic Oscillator
    A momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Stoch_k'], df['Stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=fastk_period,
                                               slowk_period=slowk_period, slowd_period=slowd_period)
    return df


def calculate_atr(df, timeperiod):
    """
    Average True Range (ATR)
    A market volatility indicator used in technical analysis, typically derived from the 14-day simple moving average of a series of true range indicators.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df


def calculate_chaikin_money_flow(df, fastperiod, slowperiod):
    """
    Chaikin Money Flow (CMF)
    A technical analysis indicator used to measure the volume-weighted average of accumulation and distribution over a specified period.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['CMF'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=fastperiod,
                            slowperiod=slowperiod)
    return df


def calculate_obv(df):
    """
    On-Balance Volume (OBV)
    A technical trading momentum indicator that uses volume flow to predict changes in stock price.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['OBV'] = talib.OBV(df['close'], df['volume'])
    return df


def calculate_ichimoku_cloud(df, conversion_line_period, base_line_period, lead_span_b_period):
    """
    Ichimoku Cloud
    A collection of technical indicators that show support and resistance levels, as well as momentum and trend direction.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    nine_period_high = df['high'].rolling(window=conversion_line_period).max()
    nine_period_low = df['low'].rolling(window=conversion_line_period).min()
    df['Tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    period26_high = df['high'].rolling(window=base_line_period).max()
    period26_low = df['low'].rolling(window=base_line_period).min()
    df['Kijun_sen'] = (period26_high + period26_low) / 2

    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(base_line_period)
    period52_high = df['high'].rolling(window=lead_span_b_period).max()
    period52_low = df['low'].rolling(window=lead_span_b_period).min()
    df['Senkou_Span_B'] = ((period52_high + period52_low) / 2).shift(lead_span_b_period)

    df['Chikou_Span'] = df['close'].shift(-base_line_period)
    return df


# Market Facilitation Index (MFI)
def calculate_mfi(df):
    """
    Market Facilitation Index (MFI)
    The MFI is an indicator that relates price and volume in the market. It is calculated as the difference between high and low divided by volume.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['MFI'] = (df['high'] - df['low']) / df['volume']
    return df


# Average True Range Percentage (ATR %)
def calculate_atr_percentage(df, timeperiod):
    """
    Average True Range Percentage (ATR %)
    ATR Percentage shows the relative level of volatility and is useful for comparing volatility across different price levels.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    df['ATR_Percentage'] = (atr / df['close']) * 100
    return df


# Heikin-Ashi Technique
def calculate_heikin_ashi(df):
    """
    Heikin-Ashi Technique
    A charting method that averages the open, close, high, and low of traditional candlesticks to create a smoother chart that highlights trends more effectively.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['HA_Open'] = ((df['open'].shift(1) + df['close'].shift(1)) / 2).fillna(df['open'].iloc[0])
    df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)
    return df


# Trend Intensity Index (TII)
def calculate_tii(df, period):
    """
    Trend Intensity Index (TII)
    TII is used to indicate the strength of a current trend in the market.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['TII'] = 100 * (df['close'] - df['close'].rolling(window=period).mean()) / df['close'].rolling(
        window=period).mean()
    return df


# Kaufman’s Adaptive Moving Average (KAMA)
def calculate_kama(df, timeperiod):
    """
    Kaufman’s Adaptive Moving Average (KAMA)
    KAMA accounts for market noise or volatility. It adjusts its smoothing to reflect the market's current volatility.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['KAMA'] = talib.KAMA(df['close'], timeperiod=timeperiod)
    return df


# Wave Trend Oscillator (WTO)
def calculate_wto(df, n1, n2):
    """
    Wave Trend Oscillator (WTO)
    A momentum indicator useful for identifying the start and reversal of trends.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    ap = (df['high'] + df['low'] + df['close']) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (np.abs(ap - esa)).ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    df['WTO'] = ci.ewm(span=n2, adjust=False).mean()
    return df


# Gann High Low Activator (HLA)
def calculate_hla(df, period):
    """
    Gann High Low Activator (HLA)
    HLA is a trend-following indicator used to identify market breakouts and reversals.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['HLA'] = (df['high'].rolling(window=period).max() + df['low'].rolling(window=period).min()) / 2
    return df


# Donchian Channels
def calculate_donchian_channels(df, period):
    """
    Donchian Channels
    A volatility indicator based on the high and low prices over a specified number of periods.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Donchian_Channel_High'] = df['high'].rolling(window=period).max()
    df['Donchian_Channel_Low'] = df['low'].rolling(window=period).min()
    return df


def calculate_ama(df, period):
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    # Calculate the Efficiency Ratio (ER)
    change = df['close'].diff().abs()
    volatility = change.rolling(window=period).sum()
    er = change.rolling(window=period).sum() / volatility

    # Constants for fast and slow EMA
    fast = 2 / (2 + 1)
    slow = 2 / (30 + 1)

    # Calculate the smoothing constant
    sc = (er * (fast - slow) + slow) ** 2

    # Replace NaN in smoothing constant with the slow constant
    sc.fillna(slow ** 2, inplace=True)

    # Initialize AMA with the first close price
    ama = pd.Series(index=df.index, dtype=float)
    ama.iloc[0] = df['close'].iloc[0]

    # Calculate AMA for each point
    for i in range(1, len(df)):
        if pd.notna(df['close'].iloc[i]):
            ama.iloc[i] = ama.iloc[i - 1] + sc.iloc[i] * (df['close'].iloc[i] - ama.iloc[i - 1])
        else:
            ama.iloc[i] = ama.iloc[i - 1]  # Handle NaN in close prices

    df['AMA'] = ama
    return df


# ZigZag Indicator
def calculate_zigzag(df, threshold_percentage):
    """
    ZigZag Indicator
    The ZigZag indicator filters out changes in an underlying plot (e.g., a price chart) that are below a specified threshold.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    threshold_value = df['close'].mean() * threshold_percentage

    # Initialize the trend with the first price
    last_pivot = df.iloc[0]['close']
    pivots = [last_pivot]
    direction = 0

    # Create a Series to hold the Zig Zag values, initialized to NaN
    zigzag_series = pd.Series(index=df.index, dtype=float)
    zigzag_series.iloc[0] = last_pivot  # Set the first value

    for i in range(1, len(df)):
        price = df.iloc[i]['close']
        if direction <= 0:  # Looking for an upward pivot
            if price >= last_pivot + threshold_value:
                direction = 1
                pivots.append(price)
                zigzag_series.iloc[i] = price
                last_pivot = price
        else:  # Looking for a downward pivot
            if price <= last_pivot - threshold_value:
                direction = -1
                pivots.append(price)
                zigzag_series.iloc[i] = price
                last_pivot = price

    df['ZigZag'] = zigzag_series
    return df


def calculate_macd(df, fastperiod, slowperiod, signalperiod):
    """
    Moving Average Convergence Divergence (MACD)
    MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = talib.MACD(df['close'], fastperiod=fastperiod,
                                                                     slowperiod=slowperiod, signalperiod=signalperiod)
    return df


def calculate_rsi(df, timeperiod):
    """
    Relative Strength Index (RSI)
    RSI is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['RSI'] = talib.RSI(df['close'], timeperiod=timeperiod)
    return df


def calculate_bollinger_bands(df, timeperiod, nbdevup, nbdevdn):
    """
    Bollinger Bands
    A technical analysis tool defined by a set of lines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of a security's price.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = talib.BBANDS(df['close'], timeperiod=timeperiod,
                                                                         nbdevup=nbdevup, nbdevdn=nbdevdn)
    return df


# Commodity Channel Index (CCI)
def calculate_cci(df, timeperiod):
    """
    Commodity Channel Index (CCI)
    An oscillator used in technical analysis to help determine when an investment vehicle has been overbought and oversold.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df


# Williams %R
def calculate_williams_r(df, timeperiod):
    """
    Williams %R
    A momentum indicator that measures overbought and oversold levels.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Williams_R'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df


# Momentum Indicator
def calculate_momentum(df, timeperiod):
    """
    Momentum Indicator
    Measures the rate at which the price of a security is changing.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Momentum'] = talib.MOM(df['close'], timeperiod=timeperiod)
    return df


# True Strength Index (TSI)
def calculate_tsi(df, high_period, low_period):
    """
    True Strength Index (TSI)
    A momentum oscillator that tracks the direction and magnitude of price changes.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    momentum = df['close'].diff()
    first_smoothing = momentum.ewm(span=high_period, adjust=False).mean()
    second_smoothing = first_smoothing.ewm(span=low_period, adjust=False).mean()
    first_smoothing_abs = momentum.abs().ewm(span=high_period, adjust=False).mean()
    second_smoothing_abs = first_smoothing_abs.ewm(span=low_period, adjust=False).mean()
    df['TSI'] = 100 * (second_smoothing / second_smoothing_abs)
    return df


# Ultimate Oscillator
def calculate_ultimate_oscillator(df, timeperiod1, timeperiod2, timeperiod3):
    """
    Ultimate Oscillator
    A technical indicator that incorporates three different timeframes to reduce the volatility and false transaction signals associated with other indicators.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Ultimate_Osc'] = talib.ULTOSC(df['high'], df['low'], df['close'],
                                      timeperiod1=timeperiod1,
                                      timeperiod2=timeperiod2,
                                      timeperiod3=timeperiod3)
    return df


# Standard Deviation
def calculate_std_dev(df, timeperiod):
    """
    Standard Deviation
    Measures the market volatility and the dispersion of prices from the mean.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Std_Dev'] = talib.STDDEV(df['close'], timeperiod=timeperiod, nbdev=1)
    return df


# Average Directional Movement Index Rating (ADXR)
def calculate_adxr(df, timeperiod):
    """
    Average Directional Movement Index Rating (ADXR)
    An indicator of trend strength in a series of prices of a financial instrument.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['ADXR'] = talib.ADXR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df


# Aroon Indicator
def calculate_aroon(df, timeperiod):
    """
    Aroon Indicator
    Designed to signal the start of a new trend or the continuation of an existing trend within price data.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Aroon_Up'], df['Aroon_Down'] = talib.AROON(df['high'], df['low'], timeperiod=timeperiod)
    return df


# Rate of Change (ROC)
def calculate_roc(df, timeperiod=10):
    """
    Rate of Change (ROC)
    A momentum oscillator that measures the percentage change between the current price and the n period past price.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['ROC'] = talib.ROC(df['close'], timeperiod=timeperiod)
    return df


# Market Sentiment Oscillator
def calculate_market_sentiment_oscillator(df, periods=14):
    """
    Market Sentiment Oscillator
    An indicator that combines price change and volume change to gauge market sentiment.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Price_Change'] = df['close'].pct_change(periods=periods)
    df['Volume_Change'] = df['volume'].pct_change(periods=periods)
    df['Market_Sentiment_Oscillator'] = (df['Price_Change'] + df['Volume_Change']) / 2
    return df


# Linear Regression Slope
def calculate_linear_regression_slope(df, timeperiod=14):
    """
    Linear Regression Slope
    An indicator that calculates the slope of the last n periods of a security's price and is used to identify trends.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Lin_Reg_Slope'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=timeperiod)
    return df


# Fisher Transform
def calculate_fisher_transform(df, lookback=10):
    """
    Fisher Transform
    A technical indicator created by John F. Ehlers that converts prices into a Gaussian normal distribution.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['median_price'] = (df['high'] + df['low']) / 2
    df['fisher_value'] = 0.0
    for i in range(len(df)):
        if i < lookback - 1:
            continue
        max_high = df['median_price'][i - lookback + 1:i + 1].max()
        min_low = df['median_price'][i - lookback + 1:i + 1].min()
        value = 0.33 * 2 * ((df.at[df.index[i], 'median_price'] - min_low) / (max_high - min_low) - 0.5) + 0.67 * df.at[
            df.index[i - 1], 'fisher_value']
        df.at[df.index[i], 'fisher_value'] = min(max(value, -0.999), 0.999)
    df['fisher'] = 0.5 * np.log((1 + df['fisher_value']) / (1 - df['fisher_value']))
    df['fisher_signal'] = df['fisher'].shift(1)
    return df


# Ehlers Fisher Transform
def calculate_ehlers_fisher_transform(df, period=10):
    """
    Ehlers Fisher Transform
    A variation of the Fisher Transform. It provides clearer turning points and overbought/oversold signals.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    median_price = (df['high'] + df['low']) / 2
    rolling_max = median_price.rolling(window=period).max()
    rolling_min = median_price.rolling(window=period).min()
    normalized_value = 0.33 * 2 * ((median_price - rolling_min) / (rolling_max - rolling_min) - 0.5)
    normalized_value = normalized_value.clip(-0.999, 0.999)
    ehlers_value = normalized_value + 0.67 * normalized_value.shift(1)
    ehlers_value.fillna(0.0, inplace=True)
    fisher_transform = 0.5 * np.log((1 + ehlers_value) / (1 - ehlers_value))
    fisher_trigger = fisher_transform.shift(1)
    df['Ehlers_Fisher'] = fisher_transform
    df['Ehlers_Trigger'] = fisher_trigger
    return df


# Elder's Force Index (EFI)
def calculate_elders_force_index(df):
    """
    Elder's Force Index (EFI)
    An oscillator that uses price and volume to assess the power behind a price move.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['EFI'] = (df['close'] - df['close'].shift(1)) * df['volume']
    return df


# Elder-Ray Index
def calculate_elder_ray_index(df, ema_period=13):
    """
    Elder-Ray Index
    Developed by Dr. Alexander Elder, this indicator measures buying and selling pressure in the market.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Elder_Bull_Power'] = df['high'] - talib.EMA(df['close'], timeperiod=ema_period)
    df['Elder_Bear_Power'] = df['low'] - talib.EMA(df['close'], timeperiod=ema_period)
    return df


# Keltner Channels
def calculate_keltner_channels(df, ema_period=20, atr_period=10, multiplier=1.5):
    """
    Keltner Channels
    A volatility based 'envelope' that can help identify overbought and oversold levels.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Keltner_Channel_Middle'] = talib.EMA(df['close'], timeperiod=ema_period)
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    df['Keltner_Channel_Upper'] = df['Keltner_Channel_Middle'] + atr * multiplier
    df['Keltner_Channel_Lower'] = df['Keltner_Channel_Middle'] - atr * multiplier
    return df


# Detrended Price Oscillator (DPO)
def calculate_dpo(df, period=20):
    """
    Detrended Price Oscillator (DPO)
    An oscillator designed to identify cycles and trends in a security's price.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    displaced_period = int((0.5 * period) + 1)
    df['DPO'] = df['close'].shift(displaced_period) - df['close'].rolling(window=period).mean()
    return df


# Accumulation/Distribution Line (A/D Line)
def calculate_ad_line(df):
    """
    Accumulation/Distribution Line (A/D Line)
    An indicator designed to measure the cumulative flow of money into and out of a security.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['A/D_Line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    return df


# Vortex Indicator (VI)
def calculate_vortex_indicator(df, period=14):
    """
    Vortex Indicator (VI)
    An indicator designed to identify the start of a new trend or the continuation of an existing trend.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['VM+'] = abs(df['high'] - df['low'].shift(1))
    df['VM-'] = abs(df['low'] - df['high'].shift(1))
    df['Sum_VM+'] = df['VM+'].rolling(window=period).sum()
    df['Sum_VM-'] = df['VM-'].rolling(window=period).sum()
    df['TR'] = talib.TRANGE(df['high'], df['low'], df['close'])
    df['Sum_TR'] = df['TR'].rolling(window=period).sum()
    df['VI+'] = df['Sum_VM+'] / df['Sum_TR']
    df['VI-'] = df['Sum_VM-'] / df['Sum_TR']
    return df


# Volume Oscillator
def calculate_volume_oscillator(df, short_period=5, long_period=10):
    """
    Volume Oscillator
    Measures the difference between two volume-based moving averages.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    short_vol = df['volume'].rolling(window=short_period).mean()
    long_vol = df['volume'].rolling(window=long_period).mean()
    df['Volume_Oscillator'] = short_vol - long_vol
    return df


# Price Oscillator
def calculate_price_oscillator(df, short_period=5, long_period=10):
    """
    Price Oscillator
    Measures the difference between two price-based moving averages.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    short_price = df['close'].rolling(window=short_period).mean()
    long_price = df['close'].rolling(window=long_period).mean()
    df['Price_Oscillator'] = (short_price - long_price) / long_price
    return df


# Balance of Power (BOP)
def calculate_balance_of_power(df):
    """
    Balance of Power (BOP)
    Measures the strength of buyers against sellers in the market.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['BOP'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    return df


def WMA(df, period):
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    return df.rolling(period).apply(lambda x: ((np.arange(period) + 1) * x).sum() / (np.arange(period) + 1).sum(),
                                    raw=True)


# Hull Moving Average (HMA)
def calculate_hma(df, period):
    """
    Hull Moving Average (HMA)
    A faster and smoother moving average, useful for identifying the current market trend.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    df['HMA'] = WMA(2 * WMA(df['close'], half_length) - WMA(df['close'], period), sqrt_length)
    return df
