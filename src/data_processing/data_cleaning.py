def drop_rows_with_nan_in_columns(df, columns):
    """
    Drops rows in the DataFrame where any of the specified columns have NaN values.

    :param df: Pandas DataFrame
    :param columns: List of columns to check for NaN values
    :return: DataFrame with rows dropped where NaN values are found in the specified columns
    """
    return df.dropna(subset=columns)


# TODO: check if all these indicators are named correctly and if in list
def remove_sparse_indicators(df):
    """
    Remove indicator columns from the DataFrame if more than 50% of their data is missing.

    :param df: pandas.DataFrame with technical indicators applied
    :return: pandas.DataFrame with sparse indicators removed
    """
    indicators_to_check = [
        'SMA_10', 'SMA_20', 'SMA_30',  # Simple Moving Averages
        'EMA_9', 'EMA_12', 'EMA_26',  # Exponential Moving Averages
        'MACD', 'MACD_signal', 'MACD_hist',  # MACD components
        'Ichimoku_Cloud_components',  # Replace with actual column names for Ichimoku Cloud components
        'CCI',  # Commodity Channel Index
        'TSI',  # True Strength Index
        'MFI',  # Money Flow Index
        'Ultimate_Oscillator',
        'Std_Dev',  # Standard Deviation
        'ADXR',  # Average Directional Movement Index Rating
        'Aroon_Up', 'Aroon_Down',  # Aroon Indicator
        'Market_Sentiment_Oscillator',
        'TRIMA',  # Triangular Moving Average
        'VAMA',  # Volatility-Adjusted Moving Average
        'Keltner_Channels',
        'Donchian_Channels',
        'Heikin_Ashi',
        'TII',  # Trend Intensity Index
        'KAMA',  # Kaufman’s Adaptive Moving Average
        'WTO',  # Wave Trend Oscillator
        'AMA',  # Adaptive Moving Average
        'VPT',  # Volume-Price Trend
        'Vortex_Indicator'
    ]

    for indicator in indicators_to_check:
        if indicator in df.columns:
            missing_data_percentage = df[indicator].isna().mean() * 100
            if missing_data_percentage > 50:
                df.drop(columns=[indicator], inplace=True)

    return df


# Example usage
df_with_indicators = apply_indicators_to_dataframe(your_dataframe)
df_cleaned = remove_sparse_indicators(df_with_indicators)

columns_to_check = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime_utc']
data_cleaned = drop_rows_with_nan_in_columns(df, columns_to_check)
