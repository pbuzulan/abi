import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Trend Intensity Index (TII)
def calculate_tii(df: pd.DataFrame, timeperiod: int):
    """
    Trend Intensity Index (TII)
    TII is used to indicate the strength of a current trend in the market.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['TII'] = 100 * (df['close'] - df['close'].rolling(window=timeperiod).mean()) / df['close'].rolling(
        window=timeperiod).mean()
    return df
