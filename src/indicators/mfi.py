import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Market Facilitation Index (MFI)
def calculate_mfi(df: pd.DataFrame):
    """
    Market Facilitation Index (MFI)
    The MFI is an indicator that relates price and volume in the market. It is calculated as the difference between high and low divided by volume.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['MFI'] = (df['high'] - df['low']) / df['volume']
    return df
