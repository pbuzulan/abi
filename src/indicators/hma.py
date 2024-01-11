import logging
import pandas as pd
import talib
import numpy as np
import inspect

logger = logging.getLogger(__name__)


def WMA(df: pd.DataFrame, timeperiod: int):
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    return df.rolling(timeperiod).apply(lambda x: ((np.arange(timeperiod) + 1) * x).sum() / (np.arange(timeperiod) + 1).sum(),
                                        raw=True)


# Hull Moving Average (HMA)
def calculate_hma(df: pd.DataFrame, timeperiod: int):
    """
    Hull Moving Average (HMA)
    A faster and smoother moving average, useful for identifying the current market trend.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    half_length = int(timeperiod / 2)
    sqrt_length = int(np.sqrt(timeperiod))
    df['HMA'] = WMA(2 * WMA(df['close'], half_length) - WMA(df['close'], timeperiod), sqrt_length)
    return df
