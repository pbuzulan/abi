import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


def calculate_bollinger_bands(df: pd.DataFrame, timeperiod: int, nbdevup: int, nbdevdn: int):
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
