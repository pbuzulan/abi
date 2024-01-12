import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


def calculate_rsi(df: pd.DataFrame, timeperiod: int):
    """
    Relative Strength Index (RSI)
    RSI is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['RSI'] = talib.RSI(df['close'], timeperiod=timeperiod)
    return df
