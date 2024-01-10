import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Donchian Channels
def calculate_donchian_channels(df: pd.DataFrame, period: int):
    """
    Donchian Channels
    A volatility indicator based on the high and low prices over a specified number of periods.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Donchian_Channel_High'] = df['high'].rolling(window=period).max()
    df['Donchian_Channel_Low'] = df['low'].rolling(window=period).min()
    return df
