import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Aroon Indicator
def calculate_aroon(df: pd.DataFrame, timeperiod: int):
    """
    Aroon Indicator
    Designed to signal the start of a new trend or the continuation of an existing trend within price data.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Aroon_Up'], df['Aroon_Down'] = talib.AROON(df['high'], df['low'], timeperiod=timeperiod)
    return df
