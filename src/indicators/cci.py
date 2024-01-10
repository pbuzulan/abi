import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Commodity Channel Index (CCI)
def calculate_cci(df: pd.DataFrame, timeperiod: int):
    """
    Commodity Channel Index (CCI)
    An oscillator used in technical analysis to help determine when an investment vehicle has been overbought and oversold.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=timeperiod)
    return df
