import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Elder-Ray Index
def calculate_elder_ray_index(df: pd.DataFrame, ema_period: int):
    """
    Elder-Ray Index
    Developed by Dr. Alexander Elder, this indicator measures buying and selling pressure in the market.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Elder_Bull_Power'] = df['high'] - talib.EMA(df['close'], timeperiod=ema_period)
    df['Elder_Bear_Power'] = df['low'] - talib.EMA(df['close'], timeperiod=ema_period)
    return df
