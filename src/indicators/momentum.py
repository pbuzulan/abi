import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Momentum Indicator
def calculate_momentum(df: pd.DataFrame, timeperiod: int):
    """
    Momentum Indicator
    Measures the rate at which the price of a security is changing.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Momentum'] = talib.MOM(df['close'], timeperiod=timeperiod)
    return df
