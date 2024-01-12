import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler


def fill_missing_values(df):
    volume_price_columns = ['open', 'high', 'low', 'close', 'volume']
    cumulative_indicators = ['A/D_Line', 'OBV', 'CMF']
    moving_averages_oscillators = ['SMA_10', 'EMA_9', 'RSI', 'MACD', 'Upper_Band', 'Middle_Band', 'Lower_Band']
    volatility_measures = ['ATR', 'Std_Dev']
    ichimoku_components = ['Tenkan_sen', 'Kijun_sen', 'Senkou_Span_A', 'Senkou_Span_B', 'Chikou_Span']
    fixed_range_indicators = ['Williams_R', 'Stoch_k', 'Stoch_d', 'MFI']
    momentum_indicators = ['Momentum', 'ROC', 'Price_Change']
    others = ['Lin_Reg_Slope', 'DPO', 'VI+', 'VI-', 'Market_Sentiment_Oscillator', 'Ultimate_Osc', 'Volume_Oscillator',
              'Ehlers_Fisher', 'Ehlers_Trigger', 'Elder_Bull_Power', 'Elder_Bear_Power', 'EFI', 'Price_Oscillator',
              'TII', 'TSI', 'KAMA', 'HLA', 'HMA', 'Keltner_Channel_Middle', 'Keltner_Channel_Upper',
              'Keltner_Channel_Lower']

    for column in df.columns:
        if column in volume_price_columns:
            continue  # Skip filling for raw volume and price data
        elif column in cumulative_indicators + moving_averages_oscillators + volatility_measures + ichimoku_components + fixed_range_indicators + momentum_indicators + others:
            df[column].fillna(method='ffill', inplace=True)  # Forward fill for these indicators
        else:
            df[column].fillna(method='ffill', inplace=True)  # Default to forward fill

    return df


def normalize_dataset(df):
    """
    Prepares the data for machine learning models by handling missing values,
    separating features, target, and identifiers, and scaling the features using RobustScaler.

    Parameters:
    df (DataFrame): The original DataFrame with all the data.

    Returns:
    DataFrame: A new DataFrame with scaled features, target, and identifiers.
    """
    # Fill missing values if any
    df = fill_missing_values(df)

    # Separate features, target, and identifiers
    features = df.drop(['datetime_utc', 'timestamp', 'close'], axis=1)
    target = df['close']
    identifiers = df[['datetime_utc', 'timestamp']]

    # Initialize the RobustScaler
    scaler = RobustScaler()

    # Fit and transform the features
    scaled_features = scaler.fit_transform(features)

    # Create a new DataFrame for scaled features
    df_scaled = pd.DataFrame(scaled_features, columns=features.columns)

    # Add the target and identifiers back to the scaled DataFrame
    df_scaled['close'] = target
    df_scaled[['datetime_utc', 'timestamp']] = identifiers

    return df_scaled
