import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


def calculate_ama(df: pd.DataFrame, timeperiod: int):
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    # Calculate the Efficiency Ratio (ER)
    change = df['close'].diff().abs()
    volatility = change.rolling(window=timeperiod).sum()
    er = change.rolling(window=timeperiod).sum() / volatility

    # Constants for fast and slow EMA
    fast = 2 / (2 + 1)
    slow = 2 / (30 + 1)

    # Calculate the smoothing constant
    sc = (er * (fast - slow) + slow) ** 2

    # Replace NaN in smoothing constant with the slow constant
    sc.fillna(slow ** 2, inplace=True)

    # Initialize AMA with the first close price
    ama = pd.Series(index=df.index, dtype=float)
    ama.iloc[0] = df['close'].iloc[0]

    # Calculate AMA for each point
    for i in range(1, len(df)):
        if pd.notna(df['close'].iloc[i]):
            ama.iloc[i] = ama.iloc[i - 1] + sc.iloc[i] * (df['close'].iloc[i] - ama.iloc[i - 1])
        else:
            ama.iloc[i] = ama.iloc[i - 1]  # Handle NaN in close prices

    df['AMA'] = ama
    return df
