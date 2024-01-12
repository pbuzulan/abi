import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


def calculate_sma(df: pd.DataFrame, timeperiod: int):
    """
    Simple Moving Average (SMA)
    SMA is a technical indicator that averages the closing prices over a specified period.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df[f'SMA_{timeperiod}'] = talib.SMA(df['close'], timeperiod=timeperiod)
    return df
