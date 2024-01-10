import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Balance of Power (BOP)
def calculate_balance_of_power(df: pd.DataFrame):
    """
    Balance of Power (BOP)
    Measures the strength of buyers against sellers in the market.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['BOP'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    return df
