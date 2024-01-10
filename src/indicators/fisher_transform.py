import logging
import pandas as pd
import talib
import numpy as np
import inspect

logger = logging.getLogger(__name__)


# Fisher Transform
def calculate_fisher_transform(df: pd.DataFrame, lookback: int):
    """
    Fisher Transform
    A technical indicator created by John F. Ehlers that converts prices into a Gaussian normal distribution.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['median_price'] = (df['high'] + df['low']) / 2
    df['fisher_value'] = 0.0
    for i in range(len(df)):
        if i < lookback - 1:
            continue
        max_high = df['median_price'][i - lookback + 1:i + 1].max()
        min_low = df['median_price'][i - lookback + 1:i + 1].min()
        value = 0.33 * 2 * ((df.at[df.index[i], 'median_price'] - min_low) / (max_high - min_low) - 0.5) + 0.67 * df.at[
            df.index[i - 1], 'fisher_value']
        df.at[df.index[i], 'fisher_value'] = min(max(value, -0.999), 0.999)
    df['fisher'] = 0.5 * np.log((1 + df['fisher_value']) / (1 - df['fisher_value']))
    df['fisher_signal'] = df['fisher'].shift(1)
    return df
