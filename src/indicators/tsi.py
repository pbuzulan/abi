import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# True Strength Index (TSI)
def calculate_tsi(df: pd.DataFrame, high_period: int, low_period: int):
    """
    True Strength Index (TSI)
    A momentum oscillator that tracks the direction and magnitude of price changes.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    momentum = df['close'].diff()
    first_smoothing = momentum.ewm(span=high_period, adjust=False).mean()
    second_smoothing = first_smoothing.ewm(span=low_period, adjust=False).mean()
    first_smoothing_abs = momentum.abs().ewm(span=high_period, adjust=False).mean()
    second_smoothing_abs = first_smoothing_abs.ewm(span=low_period, adjust=False).mean()
    df['TSI'] = 100 * (second_smoothing / second_smoothing_abs)
    return df
