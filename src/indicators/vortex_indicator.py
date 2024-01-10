import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Vortex Indicator (VI)
def calculate_vortex_indicator(df: pd.DataFrame, period: int):
    """
    Vortex Indicator (VI)
    An indicator designed to identify the start of a new trend or the continuation of an existing trend.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['VM+'] = abs(df['high'] - df['low'].shift(1))
    df['VM-'] = abs(df['low'] - df['high'].shift(1))
    df['Sum_VM+'] = df['VM+'].rolling(window=period).sum()
    df['Sum_VM-'] = df['VM-'].rolling(window=period).sum()
    df['TR'] = talib.TRANGE(df['high'], df['low'], df['close'])
    df['Sum_TR'] = df['TR'].rolling(window=period).sum()
    df['VI+'] = df['Sum_VM+'] / df['Sum_TR']
    df['VI-'] = df['Sum_VM-'] / df['Sum_TR']
    return df
