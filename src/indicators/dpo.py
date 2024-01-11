import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Detrended Price Oscillator (DPO)
def calculate_dpo(df: pd.DataFrame, timeperiod: int):
    """
    Detrended Price Oscillator (DPO)
    An oscillator designed to identify cycles and trends in a security's price.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    displaced_period = int((0.5 * timeperiod) + 1)
    df['DPO'] = df['close'].shift(displaced_period) - df['close'].rolling(window=timeperiod).mean()
    return df
