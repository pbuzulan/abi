import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Ultimate Oscillator
def calculate_ultimate_oscillator(df: pd.DataFrame, timeperiod1: int, timeperiod2: int, timeperiod3: int):
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
