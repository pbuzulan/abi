import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# TODO: figure out the numbers
def calculate_ichimoku_cloud(df: pd.DataFrame, conversion_line_period, base_line_period, lead_span_b_period):
    """
    Ichimoku Cloud
    A collection of technical indicators that show support and resistance levels, as well as momentum and trend direction.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    nine_period_high = df['high'].rolling(window=conversion_line_period).max()
    nine_period_low = df['low'].rolling(window=conversion_line_period).min()
    df['Tenkan_sen'] = (nine_period_high + nine_period_low) / 2

    period26_high = df['high'].rolling(window=base_line_period).max()
    period26_low = df['low'].rolling(window=base_line_period).min()
    df['Kijun_sen'] = (period26_high + period26_low) / 2

    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(base_line_period)
    period52_high = df['high'].rolling(window=lead_span_b_period).max()
    period52_low = df['low'].rolling(window=lead_span_b_period).min()
    df['Senkou_Span_B'] = ((period52_high + period52_low) / 2).shift(lead_span_b_period)

    df['Chikou_Span'] = df['close'].shift(-base_line_period)
    return df
