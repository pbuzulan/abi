import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Keltner Channels
def calculate_keltner_channels(df: pd.DataFrame, ema_period: int, atr_period: int, multiplier: float):
    """
    Keltner Channels
    A volatility based 'envelope' that can help identify overbought and oversold levels.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Keltner_Channel_Middle'] = talib.EMA(df['close'], timeperiod=ema_period)
    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    df['Keltner_Channel_Upper'] = df['Keltner_Channel_Middle'] + atr * multiplier
    df['Keltner_Channel_Lower'] = df['Keltner_Channel_Middle'] - atr * multiplier
    return df
