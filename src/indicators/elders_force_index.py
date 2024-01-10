import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Elder's Force Index (EFI)
def calculate_elders_force_index(df: pd.DataFrame):
    """
    Elder's Force Index (EFI)
    An oscillator that uses price and volume to assess the power behind a price move.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['EFI'] = (df['close'] - df['close'].shift(1)) * df['volume']
    return df
