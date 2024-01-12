import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Williams %R
def calculate_williams_r(df: pd.DataFrame, timeperiod: int):
    """
    Williams %R
    A momentum indicator that measures overbought and oversold levels.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Williams_R'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df
