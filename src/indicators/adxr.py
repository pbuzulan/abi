import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Average Directional Movement Index Rating (ADXR)
def calculate_adxr(df: pd.DataFrame, timeperiod: int):
    """
    Average Directional Movement Index Rating (ADXR)
    An indicator of trend strength in a series of prices of a financial instrument.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['ADXR'] = talib.ADXR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df
