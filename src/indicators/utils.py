from sklearn.preprocessing import StandardScaler
import pandas as pd


def drop_rows_with_nan_in_columns(df, columns):
    """
    Drops rows in the DataFrame where any of the specified columns have NaN values.

    :param df: Pandas DataFrame
    :param columns: List of columns to check for NaN values
    :return: DataFrame with rows dropped where NaN values are found in the specified columns
    """
    return df.dropna(subset=columns)


def normalize_dataset(df):
    """
    Prepares the data for machine learning models by handling missing values,
    separating features, target, and identifiers, and scaling the features.

    Parameters:
    df (DataFrame): The original DataFrame with all the data.

    Returns:
    DataFrame: A new DataFrame with scaled features, target, and identifiers.
    """
    # Fill missing values if any
    df.fillna(method='ffill', inplace=True)

    # Separate features, target, and identifiers
    features = df.drop(['datetime_utc', 'timestamp', 'close'],
                       axis=1)  # Exclude datetime and timestamp, and target column
    target = df['close']
    identifiers = df[['datetime_utc', 'timestamp']]

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the features
    scaled_features = scaler.fit_transform(features)

    # Create a new DataFrame for scaled features
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

    # Add the target and identifiers back to the scaled DataFrame
    df_scaled['close'] = target
    df_scaled[['datetime_utc', 'timestamp']] = identifiers

    return df_scaled


# Example usage
# Assuming you have a DataFrame named 'data'
columns_to_check = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime_utc']
data_cleaned = drop_rows_with_nan_in_columns(df, columns_to_check)
