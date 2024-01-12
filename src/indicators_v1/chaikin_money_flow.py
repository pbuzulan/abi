import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


def calculate_chaikin_money_flow(df: pd.DataFrame, fastperiod: int, slowperiod: int):
    """
    Chaikin Money Flow (CMF)
    A technical analysis indicator used to measure the volume-weighted average of accumulation and distribution over a specified period.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['CMF'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=fastperiod,
                            slowperiod=slowperiod)
    return df
