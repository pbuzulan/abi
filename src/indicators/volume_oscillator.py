import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Volume Oscillator
def calculate_volume_oscillator(df: pd.DataFrame, short_period: int, long_period: int):
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
