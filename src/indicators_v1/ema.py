import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


def calculate_ema(df: pd.DataFrame, timeperiod: int):
    """
    Exponential Moving Average (EMA)
    EMA is a type of moving average that places a greater weight and significance on the most recent data points.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df[f'EMA_{timeperiod}'] = talib.EMA(df['close'], timeperiod=timeperiod)
    return df
