def drop_rows_with_nan_in_columns(df, columns):
    """
    Drops rows in the DataFrame where any of the specified columns have NaN values.

    :param df: Pandas DataFrame
    :param columns: List of columns to check for NaN values
    :return: DataFrame with rows dropped where NaN values are found in the specified columns
    """
    return df.dropna(subset=columns)


columns_to_check = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime_utc']
data_cleaned = drop_rows_with_nan_in_columns(df, columns_to_check)
