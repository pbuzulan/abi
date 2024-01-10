import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Price Oscillator
def calculate_price_oscillator(df: pd.DataFrame, short_period: int, long_period: int):
    """
    Price Oscillator
    Measures the difference between two price-based moving averages.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    short_price = df['close'].rolling(window=short_period).mean()
    long_price = df['close'].rolling(window=long_period).mean()
    df['Price_Oscillator'] = (short_price - long_price) / long_price
    return df
