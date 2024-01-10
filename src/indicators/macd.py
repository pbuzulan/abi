import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


def calculate_macd(df: pd.DataFrame, fastperiod: int, slowperiod: int, signalperiod: int):
    """
    Moving Average Convergence Divergence (MACD)
    MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = talib.MACD(df['close'], fastperiod=fastperiod,
                                                                     slowperiod=slowperiod, signalperiod=signalperiod)
    return df
