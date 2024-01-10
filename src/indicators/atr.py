import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


def calculate_atr(df: pd.DataFrame, timeperiod: int):
    """
    Average True Range (ATR)
    A market volatility indicator used in technical analysis, typically derived from the 14-day simple moving average of a series of true range indicators.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df
