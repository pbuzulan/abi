import logging
import pandas as pd
import talib
import numpy as np
import inspect

logger = logging.getLogger(__name__)


# Ehlers Fisher Transform
def calculate_ehlers_fisher_transform(df: pd.DataFrame, timeperiod: int):
    """
    Ehlers Fisher Transform
    A variation of the Fisher Transform. It provides clearer turning points and overbought/oversold signals.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    median_price = (df['high'] + df['low']) / 2
    rolling_max = median_price.rolling(window=timeperiod).max()
    rolling_min = median_price.rolling(window=timeperiod).min()
    normalized_value = 0.33 * 2 * ((median_price - rolling_min) / (rolling_max - rolling_min) - 0.5)
    normalized_value = normalized_value.clip(-0.999, 0.999)
    ehlers_value = normalized_value + 0.67 * normalized_value.shift(1)
    ehlers_value.fillna(0.0, inplace=True)
    fisher_transform = 0.5 * np.log((1 + ehlers_value) / (1 - ehlers_value))
    fisher_trigger = fisher_transform.shift(1)
    df['Ehlers_Fisher'] = fisher_transform
    df['Ehlers_Trigger'] = fisher_trigger
    return df
