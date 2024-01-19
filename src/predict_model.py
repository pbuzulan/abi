import os
from src.indicators_v2.indicators import *
from src.models.bitcoin_trading_model import BitcoinTradingModel

DATA_RAW_DIR = os.getenv('DATA_RAW_DIR', '../data/raw/')
DATA_INTERMEDIATE_DIR = os.getenv('DATA_INTERMEDIATE_DIR', '../data/intermediate/')
DATA_PROCESSED_DIR = os.getenv('DATA_PROCESSED_DIR', '../data/processed/')
DATA_TEST_DIR = os.getenv('DATA_TEST_DIR', '../data/test/')
DATA_TRAINING_DIR = os.getenv('DATA_TRAINING_DIR', '../data/training/')
DATA_ANALYSIS_DIR = os.getenv('DATA_ANALYSIS_DIR', '../data/analysis/')
MODELS_DIR = os.getenv('MODELS_DIR', '../models/')
ANALYSIS_FILE_NAME = os.getenv('ANALYSIS_FILE_NAME', '../data/analysis/1D_BTC_Analysis_Data_BitcoinTradingModel.csv')
RAW_DATASET_FILE_NAME = os.getenv('RAW_DATASET_FILE_NAME', '../data/raw/BTC_ohlc_day.csv')
TEST_DATASET_FILE_NAME = os.getenv('TEST_DATASET_FILE_NAME', '../data/test/1D_BTC_Test_Data_BitcoinTradingModel.csv')
TRAINED_MODEL_FILE_NAME = os.getenv('TRAINED_MODEL_FILE_NAME',
                                    '../models/2014-01-31_2022-12-31_1D_BTC_ReturnIntervalPredictionBitcoinTradingModel_v1.pkl')


def load_raw_dataset_and_info():
    df = pd.read_csv(RAW_DATASET_FILE_NAME)
    _tmp = RAW_DATASET_FILE_NAME.split('raw/')[1]
    coin = _tmp.split('_ohlc_')[0]
    timespan = _tmp.split('_ohlc_')[1].replace(".csv", "")
    return {
        "df": df,
        "coin": coin,
        "timespan": "1D" if timespan == "day" else timespan.upper()
    }


def load_test_dataset_and_info():
    df = pd.read_csv(TEST_DATASET_FILE_NAME)
    return df


def determine_position(row):
    if row['return_interval'] in [4, 5, 6, 7, 9, 10]:
        return 'Long'
    elif row['return_interval'] in [-8, -7]:
        return 'Short'
    else:
        return 'Cash'


def compose_analysis_dataframe(df: pd.DataFrame, **kwargs):
    df.sort_values(by='timestamp', inplace=True)

    row_dataframe = load_raw_dataset_and_info()['df']

    row_dataframe.sort_values(by='timestamp', inplace=True)

    # Define the columns you want to replace in df1 with corresponding columns from df2
    columns_to_drop = ['open', 'high', 'low', 'close', 'volume', 'datetime_utc']

    for col in columns_to_drop:
        del df[col]

    # Merge df1 and df2 based on the 'timestamp' column
    merged_df = pd.merge(df, row_dataframe, on='timestamp', how='inner')

    merged_df['percentage_change'] = ((merged_df['close'] - merged_df['open']) / merged_df['open']) * 100

    columns_to_keep = ['actual_return_interval', 'predicted_returns', 'actual_position', 'predicted_position',
                       'percentage_change', 'open', 'high', 'low', 'close', 'volume', 'return', 'timestamp',
                       'datetime_utc']

    merged_df = merged_df[columns_to_keep]

    analysis_file_name = f'{DATA_ANALYSIS_DIR}/1D_BTC_Analysis_Data_BitcoinTradingModel.csv'

    merged_df.to_csv(analysis_file_name, index=False)

    print("analysis_file saved: ", analysis_file_name)

    return merged_df


def _categorize_return_interval(return_value):
    """
    Categorize the return value into one of the 21 predefined return intervals.

    Args:
    return_value (float): The return value to categorize.

    Returns:
    int: The category label of the return interval.
    """
    return_ranges = [
        (-100, -11), (-11, -9), (-9, -7), (-7, -5), (-5, -3), (-3, -1), (-1, -0.8), (-0.8, -0.6), (-0.6, -0.4),
        (-0.4, -0.2),
        (-0.2, 0.2),
        (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1), (1, 3), (3, 5), (5, 7), (7, 9), (9, 11), (11, float('inf'))
    ]

    for i, (lower, upper) in enumerate(return_ranges):
        if lower <= return_value < upper:
            return i - 10  # Adjust index to match the range labels from -10 to 10

    return None  # Return None if the return value does not fall into any of the ranges


def run_prediction(df: pd.DataFrame, **kwargs):
    final_df = df.copy()
    model = BitcoinTradingModel.load(kwargs['model_f_path'])

    actuals_return_interval = df['return_interval']
    actuals_returns = df['return']

    del df['return']
    del df['return_interval']
    del df['timestamp']
    del df['datetime_utc']

    metrics = model.calculate_metrics(df, actuals_return_interval, actuals_returns)

    print(metrics)

    prediction = model.predict(df)

    print(prediction)

    predicted_returns = [pred[1] for pred in prediction]
    predicted_position = [pred[0] for pred in prediction]

    final_df['actual_position'] = final_df.apply(determine_position, axis=1)

    final_df.rename(columns={'return_interval': 'actual_return_interval'}, inplace=True)

    final_df['predicted_returns'] = predicted_returns
    final_df['predicted_position'] = predicted_position

    return final_df


if __name__ == '__main__':
    # PREDICTION
    df = load_test_dataset_and_info()
    df = run_prediction(df=df, model_f_path=TRAINED_MODEL_FILE_NAME)
    df = compose_analysis_dataframe(df)
