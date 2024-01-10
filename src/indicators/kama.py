import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Kaufman’s Adaptive Moving Average (KAMA)
def calculate_kama(df: pd.DataFrame, timeperiod: int):
    """
    Kaufman’s Adaptive Moving Average (KAMA)
    KAMA accounts for market noise or volatility. It adjusts its smoothing to reflect the market's current volatility.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['KAMA'] = talib.KAMA(df['close'], timeperiod=timeperiod)
    return df
