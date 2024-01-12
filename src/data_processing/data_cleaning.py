def drop_rows_with_nan_in_columns(df):
    """
    Drop columns from a DataFrame where all values are NaN.

    :param df: pandas.DataFrame
    :return: pandas.DataFrame with empty columns dropped
    """
    return df.dropna(axis=1, how='all')


def remove_sparse_indicators(df):
    """
    Remove indicator columns from the DataFrame if more than 50% of their data is missing.

    :param df: pandas.DataFrame with technical indicators applied
    :return: pandas.DataFrame with sparse indicators removed
    """
    indicators_to_check = [
        'SMA_10', 'SMA_20', 'SMA_30',  # Simple Moving Averages
        'EMA_9', 'EMA_12', 'EMA_26',  # Exponential Moving Averages
        'MACD', 'MACD_signal', 'MACD_Histogram',  # MACD components
        'Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span',  # Ichimoku Cloud components
        'CCI',  # Commodity Channel Index
        'TSI',  # True Strength Index
        'MFI',  # Money Flow Index
        'Ultimate_Osc',
        'Std_Dev',  # Standard Deviation
        'ADXR',  # Average Directional Movement Index Rating
        'Aroon_Up', 'Aroon_Down',  # Aroon Indicator
        'Market_Sentiment_Oscillator',
        'Keltner_Channel_Middle', 'Keltner_Channel_Upper', 'Keltner_Channel_Lower',
        'Donchian_Channel_High', 'Donchian_Channel_Low',
        'HA_Close', 'HA_Open', 'HA_High', 'HA_Low',
        'TII',  # Trend Intensity Index
        'KAMA',  # Kaufman’s Adaptive Moving Average
        'WTO',  # Wave Trend Oscillator
        'AMA',  # Adaptive Moving Average
        'VM+', 'VM-', 'Sum_VM+', 'Sum_VM-', 'TR', 'Sum_TR', 'VI+', 'VI-'
    ]

    for indicator in indicators_to_check:
        if indicator in df.columns:
            missing_data_percentage = df[indicator].isna().mean() * 100
            if missing_data_percentage > 50:
                print(f"Column {indicator} has less than 50% of data available, dropping it...")
                df.drop(columns=[indicator], inplace=True)

    return df

#
# # Example usage
# import pandas as pd
#
# df = pd.read_csv('../../data/intermediate/BTC copy.csv')
#
# df = drop_rows_with_nan_in_columns(df)
# df_cleaned = remove_sparse_indicators(df)
# print(list(df_cleaned.columns))

# columns_to_check = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime_utc']
# data_cleaned = drop_rows_with_nan_in_columns(df, columns_to_check)
