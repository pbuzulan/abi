import logging
import talib
import inspect
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_bband_width(df: pd.DataFrame, timeperiod: int = 20, nbdevup: float = 2, nbdevdn: float = 2,
                          matype: int = 0) -> pd.DataFrame:
    """
    Calculate Bollinger Band Width.

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Number of periods (integer).
    nbdevup (float): Number of standard deviations for upper band.
    nbdevdn (float): Number of standard deviations for lower band.
    matype (int): Type of moving average. 0 for SMA, 1 for EMA, etc.

    Returns:
    pd.DataFrame: DataFrame with 'bband_width' column added.
    """
    upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=timeperiod, nbdevup=nbdevup,
                                                    nbdevdn=nbdevdn, matype=matype)
    df['bband_width'] = upperband - lowerband
    return df


def calculate_dema(df: pd.DataFrame, timeperiod: int = 30) -> pd.DataFrame:
    """
    Calculate Double Exponential Moving Average (DEMA).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Number of periods (integer).

    Returns:
    pd.DataFrame: DataFrame with 'dema' column added.
    """
    df['dema'] = talib.DEMA(df['close'], timeperiod=timeperiod)
    return df


def calculate_ht_trendline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Hilbert Transform - Instantaneous Trendline.

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.

    Returns:
    pd.DataFrame: DataFrame with 'ht_trendline' column added.
    """
    df['ht_trendline'] = talib.HT_TRENDLINE(df['close'])
    return df


def calculate_kama(df: pd.DataFrame, timeperiod: int = 30) -> pd.DataFrame:
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Number of periods (integer).

    Returns:
    pd.DataFrame: DataFrame with 'kama' column added.
    """
    df['kama'] = talib.KAMA(df['close'], timeperiod=timeperiod)
    return df


def calculate_midpoint(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """
    Calculate MidPoint over period.

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Number of periods (integer).

    Returns:
    pd.DataFrame: DataFrame with 'midpoint' column added.
    """
    df['midpoint'] = talib.MIDPOINT(df['close'], timeperiod=timeperiod)
    return df


def calculate_midprice(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """
    Calculate Midpoint Price over period.

    Parameters:
    df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
    timeperiod (int): Number of periods (integer).

    Returns:
    pd.DataFrame: DataFrame with 'midprice' column added.
    """
    df['midprice'] = talib.MIDPRICE(df['high'], df['low'], timeperiod=timeperiod)
    return df


def calculate_sar(df: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.DataFrame:
    """
    Calculate Parabolic SAR.

    Parameters:
    df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
    acceleration (float): Acceleration Factor.
    maximum (float): Maximum value for the acceleration factor.

    Returns:
    pd.DataFrame: DataFrame with 'sar' column added.
    """
    df['sar'] = talib.SAR(df['high'], df['low'], acceleration=acceleration, maximum=maximum)
    return df


def calculate_sarext(df: pd.DataFrame, startvalue: float = 0, offsetonreverse: float = 0,
                     accelerationinitlong: float = 0.02, accelerationlong: float = 0.02,
                     accelerationmaxlong: float = 0.2, accelerationinitshort: float = 0.02,
                     accelerationshort: float = 0.02, accelerationmaxshort: float = 0.2) -> pd.DataFrame:
    """
    Calculate Parabolic SAR - Extended.

    Parameters:
    df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
    startvalue, offsetonreverse, accelerationinitlong, accelerationlong, accelerationmaxlong,
    accelerationinitshort, accelerationshort, accelerationmaxshort (float): Parameters for SAR calculation.

    Returns:
    pd.DataFrame: DataFrame with 'sarext' column added.
    """
    df['sarext'] = talib.SAREXT(df['high'], df['low'], startvalue=startvalue, offsetonreverse=offsetonreverse,
                                accelerationinitlong=accelerationinitlong, accelerationlong=accelerationlong,
                                accelerationmaxlong=accelerationmaxlong, accelerationinitshort=accelerationinitshort,
                                accelerationshort=accelerationshort, accelerationmaxshort=accelerationmaxshort)
    return df


def calculate_tema(df: pd.DataFrame, timeperiod: int = 30) -> pd.DataFrame:
    """
    Calculate Triple Exponential Moving Average (TEMA).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Number of periods (integer).

    Returns:
    pd.DataFrame: DataFrame with 'tema' column added.
    """
    df['tema'] = talib.TEMA(df['close'], timeperiod=timeperiod)
    return df


def calculate_trima(df: pd.DataFrame, timeperiod: int = 30) -> pd.DataFrame:
    """
    Calculate Triangular Moving Average (TRIMA).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Number of periods (integer).

    Returns:
    pd.DataFrame: DataFrame with 'trima' column added.
    """
    df['trima'] = talib.TRIMA(df['close'], timeperiod=timeperiod)
    return df


def calculate_wma(df: pd.DataFrame, timeperiod: int = 30) -> pd.DataFrame:
    """
    Calculate Weighted Moving Average (WMA).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Number of periods (integer).

    Returns:
    pd.DataFrame: DataFrame with 'wma' column added.
    """
    df['wma'] = talib.WMA(df['close'], timeperiod=timeperiod)
    return df


def calculate_atr(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR).

    Parameters:
    df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.
    timeperiod (int): Number of periods (integer).

    Returns:
    pd.DataFrame: DataFrame with ATR column added.
    """
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df


def calculate_natr(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """
    Calculate Normalized Average True Range (NATR).

    Parameters:
    df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.
    timeperiod (int): Number of periods (integer).

    Returns:
    pd.DataFrame: DataFrame with NATR column added.
    """
    df['natr'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df


def calculate_trange(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate True Range (TRANGE).

    Parameters:
    df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with TRANGE column added.
    """
    df['trange'] = talib.TRANGE(df['high'], df['low'], df['close'])
    return df


def calculate_ht_dcperiod(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Hilbert Transform - Dominant Cycle Period (HT DCPERIOD).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.

    Returns:
    pd.DataFrame: DataFrame with HT DCPERIOD column added.
    """
    df['ht_dcperiod'] = talib.HT_DCPERIOD(df['close'])
    return df


def calculate_ht_dcphase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Hilbert Transform - Dominant Cycle Phase (HT DCPHASE).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.

    Returns:
    pd.DataFrame: DataFrame with HT DCPHASE column added.
    """
    df['ht_dcphase'] = talib.HT_DCPHASE(df['close'])
    return df


def calculate_ht_trendmode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Hilbert Transform - Trend vs Cycle Mode (HT TRENDMODE).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.

    Returns:
    pd.DataFrame: DataFrame with HT TRENDMODE column added.
    """
    df['ht_trendmode'] = talib.HT_TRENDMODE(df['close'])
    return df


def calculate_adx(df: pd.DataFrame, timeperiods: list = [14, 20]) -> pd.DataFrame:
    """
    Calculate Average Directional Movement Index (ADX) for different time periods.

    Parameters:
    df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
    timeperiods (list): List of integers for different time periods.

    Returns:
    pd.DataFrame: DataFrame with 'adx_[timeperiod]' columns added.
    """
    for period in timeperiods:
        df[f'adx_{period}'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
    return df


def calculate_apo(df: pd.DataFrame, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> pd.DataFrame:
    """
    Calculate Absolute Price Oscillator (APO).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    fastperiod (int): Fast EMA period.
    slowperiod (int): Slow EMA period.
    matype (int): Moving average type.

    Returns:
    pd.DataFrame: DataFrame with 'apo' column added.
    """
    df['apo'] = talib.APO(df['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
    return df


def calculate_aroonosc(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """
    Calculate Aroon Oscillator.

    Parameters:
    df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
    timeperiod (int): Time period (number of periods).

    Returns:
    pd.DataFrame: DataFrame with 'aroonosc' column added.
    """
    df['aroonosc'] = talib.AROONOSC(df['high'], df['low'], timeperiod=timeperiod)
    return df


def calculate_bop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Balance of Power (BOP).

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'bop' column added.
    """
    df['bop'] = talib.BOP(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cci(df: pd.DataFrame, timeperiods: list = [3, 5, 10, 14]) -> pd.DataFrame:
    """
    Calculate Commodity Channel Index (CCI) for different time periods.

    Parameters:
    df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
    timeperiods (list): List of integers for different time periods.

    Returns:
    pd.DataFrame: DataFrame with 'cci_[timeperiod]' columns added.
    """
    for period in timeperiods:
        df[f'cci_{period}'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=period)
    return df


def calculate_cmo(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """
    Calculate Chande Momentum Oscillator (CMO).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Time period (number of periods).

    Returns:
    pd.DataFrame: DataFrame with 'cmo' column added.
    """
    df['cmo'] = talib.CMO(df['close'], timeperiod=timeperiod)
    return df


def calculate_dx(df: pd.DataFrame, timeperiod: int = 14) -> pd.DataFrame:
    """
    Calculate Directional Movement Index (DX).

    Parameters:
    df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
    timeperiod (int): Time period (number of periods).

    Returns:
    pd.DataFrame: DataFrame with 'dx' column added.
    """
    df['dx'] = talib.DX(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df


def calculate_macd(df: pd.DataFrame, fastperiod=12, slowperiod=26, signalperiod=9) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence/Divergence (MACD).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    fastperiod (int): Fast EMA period.
    slowperiod (int): Slow EMA period.
    signalperiod (int): Signal period for MACD.

    Returns:
    pd.DataFrame: DataFrame with 'macd', 'macdsignal', and 'macdhist' columns added.
    """
    df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'], fastperiod, slowperiod, signalperiod)
    return df


def calculate_minus_di(df: pd.DataFrame, timeperiod=14) -> pd.DataFrame:
    """
    Calculate Minus Directional Indicator (-DI).

    Parameters:
    df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
    timeperiod (int): Time period (number of periods).

    Returns:
    pd.DataFrame: DataFrame with 'minus_di' column added.
    """
    df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod)
    return df


def calculate_minus_dm(df: pd.DataFrame, timeperiod=14) -> pd.DataFrame:
    """
    Calculate Minus Directional Movement (-DM).

    Parameters:
    df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
    timeperiod (int): Time period (number of periods).

    Returns:
    pd.DataFrame: DataFrame with 'minus_dm' column added.
    """
    df['minus_dm'] = talib.MINUS_DM(df['high'], df['low'], timeperiod)
    return df


def calculate_mom(df: pd.DataFrame, timeperiods: list = [1, 3, 5, 10]) -> pd.DataFrame:
    """
    Calculate Momentum (MOM) for different time periods.

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiods (list): List of integers for different time periods.

    Returns:
    pd.DataFrame: DataFrame with 'mom_[timeperiod]' columns added.
    """
    for period in timeperiods:
        df[f'mom_{period}'] = talib.MOM(df['close'], timeperiod=period)
    return df


def calculate_plus_di(df: pd.DataFrame, timeperiod=14) -> pd.DataFrame:
    """
    Calculate Plus Directional Indicator (+DI).

    Parameters:
    df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
    timeperiod (int): Time period (number of periods).

    Returns:
    pd.DataFrame: DataFrame with 'plus_di' column added.
    """
    df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod)
    return df


def calculate_plus_dm(df: pd.DataFrame, timeperiod=14) -> pd.DataFrame:
    """
    Calculate Plus Directional Movement (+DM).

    Parameters:
    df (pd.DataFrame): DataFrame with 'high' and 'low' columns.
    timeperiod (int): Time period (number of periods).

    Returns:
    pd.DataFrame: DataFrame with 'plus_dm' column added.
    """
    df['plus_dm'] = talib.PLUS_DM(df['high'], df['low'], timeperiod)
    return df


def calculate_ppo(df: pd.DataFrame, fastperiod=12, slowperiod=26, matype=0) -> pd.DataFrame:
    """
    Calculate Percentage Price Oscillator (PPO).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    fastperiod (int): Fast EMA period.
    slowperiod (int): Slow EMA period.
    matype (int): Moving average type.

    Returns:
    pd.DataFrame: DataFrame with 'ppo' column added.
    """
    df['ppo'] = talib.PPO(df['close'], fastperiod, slowperiod, matype)
    return df


def calculate_rocp(df: pd.DataFrame, timeperiod=10) -> pd.DataFrame:
    """
    Calculate Rate of Change Percentage (ROCP).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Time period.

    Returns:
    pd.DataFrame: DataFrame with 'rocp' column added.
    """
    df['rocp'] = talib.ROCP(df['close'], timeperiod)
    return df


def calculate_rocr(df: pd.DataFrame, timeperiod=10) -> pd.DataFrame:
    """
    Calculate Rate of Change Ratio (ROCR).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Time period.

    Returns:
    pd.DataFrame: DataFrame with 'rocr' column added.
    """
    df['rocr'] = talib.ROCR(df['close'], timeperiod)
    return df


def calculate_rocr100(df: pd.DataFrame, timeperiod=10) -> pd.DataFrame:
    """
    Calculate Rate of Change Ratio 100 scale (ROCR100).

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Time period.

    Returns:
    pd.DataFrame: DataFrame with 'rocr100' column added.
    """
    df['rocr100'] = talib.ROCR100(df['close'], timeperiod)
    return df


def calculate_rsi(df: pd.DataFrame, timeperiods: list = [5, 10, 14]) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI) for different time periods.

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiods (list): List of integers for different time periods.

    Returns:
    pd.DataFrame: DataFrame with 'rsi_[timeperiod]' columns added.
    """
    for period in timeperiods:
        df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
    return df


def calculate_stochastic(df: pd.DataFrame, fastk_period=5, slowk_period=3, slowd_period=3, slowk_matype=0,
                         slowd_matype=0) -> pd.DataFrame:
    """
    Calculate Stochastic indicator (SLOWK and SLOWD).

    Parameters:
    df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
    fastk_period, slowk_period, slowd_period (int): Periods for calculation.
    slowk_matype, slowd_matype (int): Moving average types.

    Returns:
    pd.DataFrame: DataFrame with 'slowk' and 'slowd' columns added.
    """
    df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'],
                                           fastk_period, slowk_period, slowk_matype,
                                           slowd_period, slowd_matype)
    return df


def calculate_stochastic_fast(df: pd.DataFrame, fastk_period=5, fastd_period=3, fastd_matype=0) -> pd.DataFrame:
    """
    Calculate Stochastic Fast indicator (FASTK and FASTD).

    Parameters:
    df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
    fastk_period, fastd_period (int): Periods for calculation.
    fastd_matype (int): Moving average type.

    Returns:
    pd.DataFrame: DataFrame with 'fastk' and 'fastd' columns added.
    """
    df['fastk'], df['fastd'] = talib.STOCHF(df['high'], df['low'], df['close'],
                                            fastk_period, fastd_period, fastd_matype)
    return df


def calculate_trix(df: pd.DataFrame, timeperiod=30) -> pd.DataFrame:
    """
    Calculate TRIX, the 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA.

    Parameters:
    df (pd.DataFrame): DataFrame with 'close' column.
    timeperiod (int): Time period for the Triple EMA.

    Returns:
    pd.DataFrame: DataFrame with 'trix' column added.
    """
    df['trix'] = talib.TRIX(df['close'], timeperiod)
    return df


def calculate_ultosc(df: pd.DataFrame, timeperiod1=7, timeperiod2=14, timeperiod3=28) -> pd.DataFrame:
    """
    Calculate Ultimate Oscillator.

    Parameters:
    df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
    timeperiod1, timeperiod2, timeperiod3 (int): Time periods for calculation.

    Returns:
    pd.DataFrame: DataFrame with 'ultosc' column added.
    """
    df['ultosc'] = talib.ULTOSC(df['high'], df['low'], df['close'],
                                timeperiod1, timeperiod2, timeperiod3)
    return df


def calculate_cdl2crows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Two Crows candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdl2crows' column added.
    """
    df['cdl2crows'] = talib.CDL2CROWS(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdl3blackcrows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Three Black Crows candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdl3blackcrows' column added.
    """
    df['cdl3blackcrows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdl3inside(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Three Inside Up/Down candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdl3inside' column added.
    """
    df['cdl3inside'] = talib.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdl3linestrike(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Three-Line Strike candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdl3linestrike' column added.
    """
    df['cdl3linestrike'] = talib.CDL3LINESTRIKE(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdl3outside(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Three Outside Up/Down candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdl3outside' column added.
    """
    df['cdl3outside'] = talib.CDL3OUTSIDE(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdl3starsinsouth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Three Stars in the South candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdl3starsinsouth' column added.
    """
    df['cdl3starsinsouth'] = talib.CDL3STARSINSOUTH(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdl3whitesoldiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Three Advancing White Soldiers candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdl3whitesoldiers' column added.
    """
    df['cdl3whitesoldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlabandonedbaby(df: pd.DataFrame, penetration: float = 0) -> pd.DataFrame:
    """
    Calculate Abandoned Baby candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.
    penetration (float): Penetration level (used in some candlestick pattern recognition).

    Returns:
    pd.DataFrame: DataFrame with 'cdlabandonedbaby' column added.
    """
    df['cdlabandonedbaby'] = talib.CDLABANDONEDBABY(df['open'], df['high'], df['low'], df['close'], penetration)
    return df


def calculate_cdladvanceblock(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Advance Block candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdladvanceblock' column added.
    """
    df['cdladvanceblock'] = talib.CDLADVANCEBLOCK(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlbelthold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Belt-hold candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlbelthold' column added.
    """
    df['cdlbelthold'] = talib.CDLBELTHOLD(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlbreakaway(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Breakaway candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlbreakaway' column added.
    """
    df['cdlbreakaway'] = talib.CDLBREAKAWAY(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlclosingmarubozu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Closing Marubozu candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlclosingmarubozu' column added.
    """
    df['cdlclosingmarubozu'] = talib.CDLCLOSINGMARUBOZU(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlconcealbabyswall(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Concealing Baby Swallow candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlconcealbabyswall' column added.
    """
    df['cdlconcealbabyswall'] = talib.CDLCONCEALBABYSWALL(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlcounterattack(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Counterattack candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlcounterattack' column added.
    """
    df['cdlcounterattack'] = talib.CDLCOUNTERATTACK(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdldarkcloudcover(df: pd.DataFrame, penetration: float = 0) -> pd.DataFrame:
    """
    Calculate Dark Cloud Cover candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.
    penetration (float): Penetration level (used in some candlestick pattern recognition).

    Returns:
    pd.DataFrame: DataFrame with 'cdldarkcloudcover' column added.
    """
    df['cdldarkcloudcover'] = talib.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'], penetration)
    return df


def calculate_cdldoji(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Doji candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdldoji' column added.
    """
    df['cdldoji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdldojistar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Doji Star candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdldojistar' column added.
    """
    df['cdldojistar'] = talib.CDLDOJISTAR(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdldragonflydoji(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Dragonfly Doji candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdldragonflydoji' column added.
    """
    df['cdldragonflydoji'] = talib.CDLDRAGONFLYDOJI(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlengulfing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Engulfing Pattern candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlengulfing' column added.
    """
    df['cdlengulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdleveningdojistar(df: pd.DataFrame, penetration=0) -> pd.DataFrame:
    """
    Calculate Evening Doji Star candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.
    penetration (float): Penetration level (used in some candlestick pattern recognition).

    Returns:
    pd.DataFrame: DataFrame with 'cdleveningdojistar' column added.
    """
    df['cdleveningdojistar'] = talib.CDLEVENINGDOJISTAR(df['open'], df['high'], df['low'], df['close'], penetration)
    return df


def calculate_cdleveningstar(df: pd.DataFrame, penetration=0) -> pd.DataFrame:
    """
    Calculate Evening Star candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.
    penetration (float): Penetration level (used in some candlestick pattern recognition).

    Returns:
    pd.DataFrame: DataFrame with 'cdleveningstar' column added.
    """
    df['cdleveningstar'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'], penetration)
    return df


def calculate_cdlgapsidesidewhite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Up/Down-gap side-by-side white lines candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlgapsidesidewhite' column added.
    """
    df['cdlgapsidesidewhite'] = talib.CDLGAPSIDESIDEWHITE(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlgravestonedoji(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Gravestone Doji candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlgravestonedoji' column added.
    """
    df['cdlgravestonedoji'] = talib.CDLGRAVESTONEDOJI(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlhammer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Hammer candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlhammer' column added.
    """
    df['cdlhammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlhangingman(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Hanging Man candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlhangingman' column added.
    """
    df['cdlhangingman'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlharami(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Harami Pattern candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlharami' column added.
    """
    df['cdlharami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlharamicross(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Harami Cross Pattern candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlharamicross' column added.
    """
    df['cdlharamicross'] = talib.CDLHARAMICROSS(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlhighwave(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate High-Wave Candle candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlhighwave' column added.
    """
    df['cdlhighwave'] = talib.CDLHIGHWAVE(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlhikkake(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Hikkake Pattern candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlhikkake' column added.
    """
    df['cdlhikkake'] = talib.CDLHIKKAKE(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlhikkakemod(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Modified Hikkake Pattern candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlhikkakemod' column added.
    """
    df['cdlhikkakemod'] = talib.CDLHIKKAKEMOD(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlhomingpigeon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Homing Pigeon candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlhomingpigeon' column added.
    """
    df['cdlhomingpigeon'] = talib.CDLHOMINGPIGEON(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlidentical3crows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Identical Three Crows candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlidentical3crows' column added.
    """
    df['cdlidentical3crows'] = talib.CDLIDENTICAL3CROWS(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlinneck(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate In-Neck Pattern candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlinneck' column added.
    """
    df['cdlinneck'] = talib.CDLINNECK(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlinvertedhammer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Inverted Hammer candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlinvertedhammer' column added.
    """
    df['cdlinvertedhammer'] = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlkicking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Kicking candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlkicking' column added.
    """
    df['cdlkicking'] = talib.CDLKICKING(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlkickingbylength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Kicking - bull/bear determined by the longer marubozu candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlkickingbylength' column added.
    """
    df['cdlkickingbylength'] = talib.CDLKICKINGBYLENGTH(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlladderbottom(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Ladder Bottom candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlladderbottom' column added.
    """
    df['cdlladderbottom'] = talib.CDLLADDERBOTTOM(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdllongleggeddoji(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Long Legged Doji candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdllongleggeddoji' column added.
    """
    df['cdllongleggeddoji'] = talib.CDLLONGLEGGEDDOJI(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdllongline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Long Line Candle candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdllongline' column added.
    """
    df['cdllongline'] = talib.CDLLONGLINE(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlmarubozu(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Marubozu candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlmarubozu' column added.
    """
    df['cdlmarubozu'] = talib.CDLMARUBOZU(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlmatchinglow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Matching Low candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlmatchinglow' column added.
    """
    df['cdlmatchinglow'] = talib.CDLMATCHINGLOW(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlmathold(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Mat Hold candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlmathold' column added.
    """
    df['cdlmathold'] = talib.CDLMATHOLD(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlmorningdojistar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Morning Doji Star candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlmorningdojistar' column added.
    """
    df['cdlmorningdojistar'] = talib.CDLMORNINGDOJISTAR(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlmorningstar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Morning Star candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlmorningstar' column added.
    """
    df['cdlmorningstar'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlonneck(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate On-Neck Pattern candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlonneck' column added.
    """
    df['cdlonneck'] = talib.CDLONNECK(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlpiercing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Piercing Pattern candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlpiercing' column added.
    """
    df['cdlpiercing'] = talib.CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlrickshawman(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Rickshaw Man candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlrickshawman' column added.
    """
    df['cdlrickshawman'] = talib.CDLRICKSHAWMAN(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlrisefall3methods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Rising/Falling Three Methods candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlrisefall3methods' column added.
    """
    df['cdlrisefall3methods'] = talib.CDLRISEFALL3METHODS(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlseparatinglines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Separating Lines candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlseparatinglines' column added.
    """
    df['cdlseparatinglines'] = talib.CDLSEPARATINGLINES(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlshootingstar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Shooting Star candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlshootingstar' column added.
    """
    df['cdlshootingstar'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlshortline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Short Line Candle candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlshortline' column added.
    """
    df['cdlshortline'] = talib.CDLSHORTLINE(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlspinningtop(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Spinning Top candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlspinningtop' column added.
    """
    df['cdlspinningtop'] = talib.CDLSPINNINGTOP(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlstalledpattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Stalled Pattern candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlstalledpattern' column added.
    """
    df['cdlstalledpattern'] = talib.CDLSTALLEDPATTERN(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlsticksandwich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Stick Sandwich candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlsticksandwich' column added.
    """
    df['cdlsticksandwich'] = talib.CDLSTICKSANDWICH(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdltakuri(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Takuri (Dragonfly Doji with very long lower shadow) candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdltakuri' column added.
    """
    df['cdltakuri'] = talib.CDLTAKURI(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdltasukigap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Tasuki Gap candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdltasukigap' column added.
    """
    df['cdltasukigap'] = talib.CDLTASUKIGAP(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlthrusting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Thrusting Pattern candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlthrusting' column added.
    """
    df['cdlthrusting'] = talib.CDLTHRUSTING(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdltristar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Tristar Pattern candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdltristar' column added.
    """
    df['cdltristar'] = talib.CDLTRISTAR(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlunique3river(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Unique 3 River candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlunique3river' column added.
    """
    df['cdlunique3river'] = talib.CDLUNIQUE3RIVER(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlupsidegap2crows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Upside Gap Two Crows candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlupsidegap2crows' column added.
    """
    df['cdlupsidegap2crows'] = talib.CDLUPSIDEGAP2CROWS(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_cdlxsidegap3methods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Upside/Downside Gap Three Methods candlestick pattern.

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'cdlxsidegap3methods' column added.
    """
    df['cdlxsidegap3methods'] = talib.CDLXSIDEGAP3METHODS(df['open'], df['high'], df['low'], df['close'])
    return df


def calculate_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns in percentage

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'return' column added.
    """
    df['return'] = df['close'].pct_change() * 100  # Calculate daily returns in percentage

    return df


def _categorize_return_interval(return_value):
    """
    Categorize the return value into one of the 21 predefined return intervals.

    Args:
    return_value (float): The return value to categorize.

    Returns:
    int: The category label of the return interval.
    """
    return_ranges = [
        (-100, -11), (-11, -9), (-9, -7), (-7, -5), (-5, -3), (-3, -1), (-1, -0.8), (-0.8, -0.6), (-0.6, -0.4),
        (-0.4, -0.2),
        (-0.2, 0.2),
        (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1), (1, 3), (3, 5), (5, 7), (7, 9), (9, 11), (11, float('inf'))
    ]

    for i, (lower, upper) in enumerate(return_ranges):
        if lower <= return_value < upper:
            return i - 10  # Adjust index to match the range labels from -10 to 10

    return None  # Return None if the return value does not fall into any of the ranges


def calculate_return_interval(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns in percentage

    Parameters:
    df (pd.DataFrame): DataFrame with 'open', 'high', 'low', and 'close' columns.

    Returns:
    pd.DataFrame: DataFrame with 'return_interval' column added.
    """
    df['return_interval'] = df['return'].apply(_categorize_return_interval)

    return df
