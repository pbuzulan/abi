import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Gann High Low Activator (HLA)
def calculate_hla(df: pd.DataFrame, period: int):
    """
    Gann High Low Activator (HLA)
    HLA is a trend-following indicator used to identify market breakouts and reversals.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['HLA'] = (df['high'].rolling(window=period).max() + df['low'].rolling(window=period).min()) / 2
    return df
