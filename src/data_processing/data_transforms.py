import pandas as pd
from sklearn.preprocessing import StandardScaler


# Additional Considerations:
# 1. Normalize Data: Depending on your ML model, you might need to normalize or scale the data.
# 2. Feature Selection: Use feature importance techniques to select the most relevant indicators.
# 4. Timeframe Consistency: Make sure the timeframes of your indicators match your trading strategy.


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
