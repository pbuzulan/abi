import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


def calculate_stochastic_oscillator(df: pd.DataFrame, fastk_period: int, slowk_period: int, slowd_period: int):
    """
    Stochastic Oscillator
    A momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Stoch_k'], df['Stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=fastk_period,
                                               slowk_period=slowk_period, slowd_period=slowd_period)
    return df
