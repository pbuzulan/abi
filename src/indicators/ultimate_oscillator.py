import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Ultimate Oscillator
def calculate_ultimate_oscillator(df: pd.DataFrame, period1: int, period2: int, period3: int):
    """
    Ultimate Oscillator
    A technical indicator that incorporates three different timeframes to reduce the volatility and false transaction signals associated with other indicators.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Ultimate_Osc'] = talib.ULTOSC(df['high'], df['low'], df['close'],
                                      timeperiod1=period1,
                                      timeperiod2=period2,
                                      timeperiod3=period3)
    return df
