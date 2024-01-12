import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Linear Regression Slope
def calculate_linear_regression_slope(df: pd.DataFrame, timeperiod: int):
    """
    Linear Regression Slope
    An indicator that calculates the slope of the last n periods of a security's price and is used to identify trends.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Lin_Reg_Slope'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=timeperiod)
    return df
