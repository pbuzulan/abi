import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Average True Range Percentage (ATR %)
def calculate_atr_percentage(df: pd.DataFrame, timeperiod: int):
    """
    Average True Range Percentage (ATR %)
    ATR Percentage shows the relative level of volatility and is useful for comparing volatility across different price levels.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    df['ATR_Percentage'] = (atr / df['close']) * 100
    return df
