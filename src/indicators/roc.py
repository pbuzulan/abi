import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Rate of Change (ROC)
def calculate_roc(df: pd.DataFrame, timeperiod: int):
    """
    Rate of Change (ROC)
    A momentum oscillator that measures the percentage change between the current price and the n period past price.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['ROC'] = talib.ROC(df['close'], timeperiod=timeperiod)
    return df
