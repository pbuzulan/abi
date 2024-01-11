import logging
import pandas as pd
import talib
import inspect

logger = logging.getLogger(__name__)


# Market Sentiment Oscillator
def calculate_market_sentiment_oscillator(df: pd.DataFrame, timeperiod: int):
    """
    Market Sentiment Oscillator
    An indicator that combines price change and volume change to gauge market sentiment.
    """
    args = inspect.getargvalues(inspect.currentframe()).locals
    del args['df']
    print(f"{inspect.currentframe().f_code.co_name}() executed with values: {args}")

    df['Price_Change'] = df['close'].pct_change(periods=timeperiod)
    df['Volume_Change'] = df['volume'].pct_change(periods=timeperiod)
    df['Market_Sentiment_Oscillator'] = (df['Price_Change'] + df['Volume_Change']) / 2
    return df
