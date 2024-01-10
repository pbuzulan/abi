import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Standard Deviation
def calculate_std_dev(df: pd.DataFrame, timeperiod: int):
    """
    Standard Deviation
    Measures the market volatility and the dispersion of prices from the mean.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Std_Dev'] = talib.STDDEV(df['close'], timeperiod=timeperiod, nbdev=1)
    return df
