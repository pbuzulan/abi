import logging
import pandas as pd
import talib
import numpy as np
import inspect

logger = logging.getLogger(__name__)


def WMA(df: pd.DataFrame, period: int):
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    return df.rolling(period).apply(lambda x: ((np.arange(period) + 1) * x).sum() / (np.arange(period) + 1).sum(),
                                    raw=True)


# Hull Moving Average (HMA)
def calculate_hma(df: pd.DataFrame, period: int):
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
