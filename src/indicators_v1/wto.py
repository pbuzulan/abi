import logging
import pandas as pd
import talib
import numpy as np
import inspect

logger = logging.getLogger(__name__)


# Wave Trend Oscillator (WTO)
def calculate_wto(df: pd.DataFrame, n1: int, n2: int):
    """
    Wave Trend Oscillator (WTO)
    A momentum indicator useful for identifying the start and reversal of trends.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    ap = (df['high'] + df['low'] + df['close']) / 3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (np.abs(ap - esa)).ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    df['WTO'] = ci.ewm(span=n2, adjust=False).mean()
    return df
