import os
import pandas as pd
import numpy as np

from src.analysis.data_analysis import simulate_return
from src.data_processing.data_cleaning import drop_rows_with_nan_in_columns, \
    remove_rows_with_nulls_until_full
from src.data_processing.data_transforms import normalize_dataset_v2
from src.indicators_v2.indicators import *
from src.models.bitcoin_trading_model import BitcoinTradingModel
from src.utils.files_helpers import list_files_in_directory

DATA_RAW_DIR = '../data/raw/'
DATA_INTERMEDIATE_DIR = '../data/intermediate/'
DATA_PROCESSED_DIR = '../data/processed/'
DATA_TEST_DIR = '../data/test/'
DATA_TRAINING_DIR = '../data/training/'


def load_raw_dataset_and_info():
    _file = list_files_in_directory(DATA_RAW_DIR)[0]
    df = pd.read_csv(_file)
    _tmp = _file.split('raw/')[1]
    coin = _tmp.split('_ohlc_')[0]
    timespan = _tmp.split('_ohlc_')[1].replace(".csv", "")
    return {
        "df": df,
        "coin": coin,
        "timespan": "1D" if timespan == "day" else timespan.upper()
    }


def load_test_dataset_and_info():
    _file = list_files_in_directory(DATA_TEST_DIR)[0]
    df = pd.read_csv(_file)
    return df


def determine_position(row):
    if row['return_interval'] in [4, 5, 6, 7, 9, 8, 10]:
        return 'Long'
    elif row['return_interval'] in [-8, -7]:
        return 'Short'
    else:
        return 'Cash'


def _train_model(df: pd.DataFrame, **kwargs):
    # Sort the DataFrame by 'datetime_utc' column in ascending order (oldest to newest)
    df.sort_values(by='datetime_utc', inplace=True)

    # Extract the start_date (first row) and end_date (last row)
    start_date = df['datetime_utc'].iloc[0]
    end_date = df['datetime_utc'].iloc[-1]

    del df['return']
    del df['timestamp']
    del df['datetime_utc']

    model = BitcoinTradingModel()
    model.train(data=df)

    if kwargs['save_trained_model']:
        model.save(
            f'/Users/Petru_Buzulan/Private/bi/workspace/abi/models/{start_date}_{end_date}_{kwargs["timespan"]}_{kwargs["coin"]}_{model.model_file_name}')
    return model


def split_dataset(df: pd.DataFrame, **kwargs):
    # split dataset into training set and test set
    split_timestamp = kwargs['split_timestamp']
    train_df = df[df['timestamp'] < split_timestamp]
    test_df = df[df['timestamp'] >= split_timestamp]

    if kwargs['save_datasets']:
        train_df.to_csv(
            f'{DATA_TRAINING_DIR}/{kwargs["timespan"]}_{kwargs["coin"]}_Training_Data_BitcoinTradingModel.csv',
            index=False)
        test_df.to_csv(f'{DATA_TEST_DIR}/{kwargs["timespan"]}_{kwargs["coin"]}_Test_Data_BitcoinTradingModel.csv',
                       index=False)

    return train_df, test_df


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

    columns_to_keep = ['actual_return_interval', 'predicted_returns', 'predicted_position', 'actual_position',
                       'percentage_change', 'open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime_utc']

    merged_df = merged_df[columns_to_keep]

    merged_df.to_csv('../data/analysis/actuals_vs_pred_v8.csv', index=False)

    return merged_df


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

    shifted_predicted_returns = pd.Series(predicted_returns).shift(1)
    shifted_predicted_position = pd.Series(predicted_position).shift(1)

    final_df['actual_position'] = final_df.apply(determine_position, axis=1)

    final_df.rename(columns={'return_interval': 'actual_return_interval'}, inplace=True)

    final_df['predicted_returns'] = predicted_returns
    final_df['predicted_position'] = predicted_position

    return final_df


if __name__ == '__main__':
    # PREDICTION
    model_f_path = '../models/2014-01-31_2022-12-31_1D_BTC_ReturnIntervalPredictionBitcoinTradingModel_v1.pkl'
    df = load_test_dataset_and_info()
    df = run_prediction(df=df, model_f_path=model_f_path)
    df = compose_analysis_dataframe(df)
