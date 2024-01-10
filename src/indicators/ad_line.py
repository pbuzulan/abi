import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Accumulation/Distribution Line (A/D Line)
def calculate_ad_line(df: pd.DataFrame):
    """
    Accumulation/Distribution Line (A/D Line)
    An indicator designed to measure the cumulative flow of money into and out of a security.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['A/D_Line'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
    return df
