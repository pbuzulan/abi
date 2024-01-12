import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# ZigZag Indicator
def calculate_zigzag(df: pd.DataFrame, threshold_percentage: float):
    """
    ZigZag Indicator
    The ZigZag indicator filters out changes in an underlying plot (e.g., a price chart) that are below a specified threshold.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    threshold_value = df['close'].mean() * threshold_percentage

    # Initialize the trend with the first price
    last_pivot = df.iloc[0]['close']
    pivots = [last_pivot]
    direction = 0

    # Create a Series to hold the Zig Zag values, initialized to NaN
    zigzag_series = pd.Series(index=df.index, dtype=float)
    zigzag_series.iloc[0] = last_pivot  # Set the first value

    for i in range(1, len(df)):
        price = df.iloc[i]['close']
        if direction <= 0:  # Looking for an upward pivot
            if price >= last_pivot + threshold_value:
                direction = 1
                pivots.append(price)
                zigzag_series.iloc[i] = price
                last_pivot = price
        else:  # Looking for a downward pivot
            if price <= last_pivot - threshold_value:
                direction = -1
                pivots.append(price)
                zigzag_series.iloc[i] = price
                last_pivot = price

    df['ZigZag'] = zigzag_series
    return df
