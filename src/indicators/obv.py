import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


def calculate_obv(df: pd.DataFrame):
    """
    On-Balance Volume (OBV)
    A technical trading momentum indicator that uses volume flow to predict changes in stock price.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['OBV'] = talib.OBV(df['close'], df['volume'])
    return df
