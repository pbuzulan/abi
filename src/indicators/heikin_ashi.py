import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Heikin-Ashi Technique
def calculate_heikin_ashi(df: pd.DataFrame):
    """
    Heikin-Ashi Technique
    A charting method that averages the open, close, high, and low of traditional candlesticks to create a smoother chart that highlights trends more effectively.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['HA_Open'] = ((df['open'].shift(1) + df['close'].shift(1)) / 2).fillna(df['open'].iloc[0])
    df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)
    return df
