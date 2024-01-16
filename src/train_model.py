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


def apply_indicators_to_dataframe(df: pd.DataFrame, timespan: str = '1d'):
    """
    Apply a series of technical indicators to the given DataFrame.

    :param df: pandas.DataFrame with columns 'open', 'high', 'low', 'close', 'volume'
    :param timespan: str with values '1h', '4h', '1d', '1w'
    :return: pandas.DataFrame with additional columns for each technical indicator
    """
    df = calculate_bband_width(df=df, timeperiod=20, nbdevup=2, nbdevdn=2)

    df = calculate_dema(df=df, timeperiod=20)

    df = calculate_ht_trendline(df=df)

    df = calculate_kama(df=df, timeperiod=30)

    df = calculate_midpoint(df=df, timeperiod=14)

    df = calculate_midprice(df=df, timeperiod=14)

    df = calculate_sar(df=df, acceleration=0.02, maximum=0.2)

    df = calculate_sarext(
        df=df,
        startvalue=0,
        offsetonreverse=0,
        accelerationinitlong=0.02,
        accelerationlong=0.02,
        accelerationmaxlong=0.2,
        accelerationinitshort=0.02,
        accelerationshort=0.02,
        accelerationmaxshort=0.2
    )

    df = calculate_tema(df=df, timeperiod=30)

    df = calculate_trima(df=df, timeperiod=30)

    df = calculate_wma(df=df, timeperiod=30)

    df = calculate_atr(df=df, timeperiod=14)

    df = calculate_natr(df=df, timeperiod=14)

    df = calculate_trange(df=df)

    df = calculate_ht_dcperiod(df=df)

    df = calculate_ht_dcphase(df=df)

    df = calculate_ht_trendmode(df=df)

    df = calculate_adx(df=df, timeperiods=[14, 20])

    df = calculate_apo(df=df, fastperiod=12, slowperiod=26, matype=0)

    df = calculate_aroonosc(df=df, timeperiod=14)

    df = calculate_bop(df=df)

    df = calculate_cci(df=df, timeperiods=[3, 5, 10, 14])

    df = calculate_cmo(df=df, timeperiod=14)

    df = calculate_dx(df=df, timeperiod=14)

    df = calculate_macd(df=df, fastperiod=12, slowperiod=26, signalperiod=9)

    df = calculate_minus_di(df=df, timeperiod=14)

    df = calculate_minus_dm(df=df, timeperiod=14)

    df = calculate_mom(df=df, timeperiods=[1, 3, 5, 10])

    df = calculate_plus_di(df=df, timeperiod=14)

    df = calculate_plus_dm(df=df, timeperiod=14)

    df = calculate_ppo(df=df, fastperiod=12, slowperiod=26, matype=0)

    df = calculate_rocp(df=df, timeperiod=10)

    df = calculate_rocr(df=df, timeperiod=10)

    df = calculate_rocr100(df=df, timeperiod=10)

    df = calculate_rsi(df=df, timeperiods=[5, 10, 14])

    df = calculate_stochastic(
        df=df,
        fastk_period=5,
        slowk_period=3,
        slowd_period=3,
        slowk_matype=0,
        slowd_matype=0
    )

    df = calculate_stochastic_fast(
        df=df,
        fastk_period=5,
        fastd_period=3,
        fastd_matype=0
    )

    df = calculate_trix(
        df=df, timeperiod=30
    )

    df = calculate_ultosc(df=df, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    df = calculate_cdl2crows(df=df)

    df = calculate_cdl3blackcrows(df=df)

    df = calculate_cdl3inside(df=df)

    df = calculate_cdl3linestrike(df=df)

    df = calculate_cdl3outside(df=df)

    df = calculate_cdl3starsinsouth(df=df)

    df = calculate_cdl3whitesoldiers(df=df)

    df = calculate_cdlabandonedbaby(df=df)

    df = calculate_cdladvanceblock(df=df)

    df = calculate_cdlbelthold(df=df)

    df = calculate_cdlbreakaway(df=df)

    df = calculate_cdlclosingmarubozu(df=df)

    df = calculate_cdlconcealbabyswall(df=df)

    df = calculate_cdlcounterattack(df=df)

    df = calculate_cdldarkcloudcover(df=df, penetration=0)

    df = calculate_cdldoji(df=df)

    df = calculate_cdldojistar(df=df)

    df = calculate_cdldragonflydoji(df=df)

    df = calculate_cdlengulfing(df=df)

    df = calculate_cdleveningdojistar(df=df, penetration=0)

    df = calculate_cdleveningstar(df=df, penetration=0)

    df = calculate_cdlgapsidesidewhite(df=df)

    df = calculate_cdlgravestonedoji(df=df)

    df = calculate_cdlhammer(df=df)

    df = calculate_cdlhangingman(df=df)

    df = calculate_cdlharami(df=df)

    df = calculate_cdlharamicross(df=df)

    df = calculate_cdlhighwave(df=df)

    df = calculate_cdlhikkake(df=df)

    df = calculate_cdlhikkakemod(df=df)

    df = calculate_cdlhomingpigeon(df=df)

    df = calculate_cdlidentical3crows(df=df)

    df = calculate_cdlinneck(df=df)

    df = calculate_cdlinvertedhammer(df=df)

    df = calculate_cdlkicking(df=df)

    df = calculate_cdlkickingbylength(df=df)

    df = calculate_cdlladderbottom(df=df)

    df = calculate_cdllongleggeddoji(df=df)

    df = calculate_cdllongline(df=df)

    df = calculate_cdlmarubozu(df=df)

    df = calculate_cdlmatchinglow(df=df)

    df = calculate_cdlmathold(df=df)

    df = calculate_cdlmorningdojistar(df=df)

    df = calculate_cdlmorningstar(df=df)

    df = calculate_cdlonneck(df=df)

    df = calculate_cdlpiercing(df=df)
    df = calculate_cdlrickshawman(df=df)

    df = calculate_cdlrisefall3methods(df=df)

    df = calculate_cdlseparatinglines(df=df)

    df = calculate_cdlshootingstar(df=df)

    df = calculate_cdlshortline(df=df)

    df = calculate_cdlspinningtop(df=df)

    df = calculate_cdlstalledpattern(df=df)

    df = calculate_cdlsticksandwich(df=df)

    df = calculate_cdltakuri(df=df)

    df = calculate_cdltasukigap(df=df)

    df = calculate_cdlthrusting(df=df)

    df = calculate_cdltristar(df=df)

    df = calculate_cdlunique3river(df=df)

    df = calculate_cdlupsidegap2crows(df=df)

    df = calculate_cdlxsidegap3methods(df=df)

    df = calculate_return(df=df)

    df = calculate_actual_return_interval(df=df)

    return df


def determine_position(row):
    if row['return_interval'] in [4, 5, 6, 7, 9, 8, 10]:
        return 'Long'
    elif row['return_interval'] in [-8, -7]:
        return 'Short'
    else:
        return 'Cash'


def prepare_dataset(df: pd.DataFrame):
    df = drop_rows_with_nan_in_columns(df)
    df = remove_rows_with_nulls_until_full(df)
    df = normalize_dataset_v2(df, ['return_interval', 'timestamp', 'datetime_utc'])
    return df


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


def run_training(df: pd.DataFrame, **kwargs):
    df = apply_indicators_to_dataframe(df)
    df = prepare_dataset(df)
    train_df, test_df = split_dataset(df, **kwargs)
    _train_model(train_df, **kwargs)


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

    # TODO: checking if it's correct to shift or now
    shifted_predicted_returns = pd.Series(predicted_returns).shift(1)
    shifted_predicted_position = pd.Series(predicted_position).shift(1)

    final_df['actual_position'] = final_df.apply(determine_position, axis=1)

    final_df.rename(columns={'return_interval': 'actual_return_interval'}, inplace=True)

    final_df['predicted_returns'] = predicted_returns
    final_df['predicted_position'] = predicted_position

    return final_df


if __name__ == '__main__':
    # TRAINING
    _dataset_info = load_raw_dataset_and_info()
    df = _dataset_info['df']
    coin = _dataset_info['coin']
    timespan = _dataset_info['timespan']
    run_training(
        df=df,
        split_timestamp=1672531200000,
        coin=coin,
        timespan=timespan,
        save_datasets=True,
        save_trained_model=True
    )

    # print(check_liquidation(_df))
    #
    # leverage = 10
    # maintenance_margin_rate = 0.005
    # fees_rate = 0.002

    # _df['liquidation_price'] = _df.apply(calculate_liquidation_prices,
    #                                    args=(leverage, maintenance_margin_rate, fees_rate), axis=1)

    # _df.to_csv('../data/analysis/actuals_vs_pred_v4.csv', index=False)

    # df = pd.read_csv(DATA_RAW_DIR + 'BTC_ohlc_day.csv')
    #
    #
    # df = drop_rows_with_nan_in_columns(df)
    # df = remove_sparse_indicators(df)
    #
    # df.to_csv(f'{DATA_PROCESSED_DIR}/BTC_ML_NOT_NORMALIZED.csv', index=False)
    #
    # df = normalize_dataset_v2(df, ['return_interval', 'timestamp', 'datetime_utc'])
    #
    # df.to_csv(f'{DATA_PROCESSED_DIR}/BTC_ML_NORMALIZED.csv', index=False)

    # df = pd.read_csv(DATA_PROCESSED_DIR + 'BTC_ML_NORMALIZED.csv')
    #
    # split_timestamp = 1672531200000
    # train_df = df[df['timestamp'] < split_timestamp]
    # test_df = df[df['timestamp'] >= split_timestamp]
    #
    # del train_df['return']
    # del train_df['timestamp']
    # del train_df['datetime_utc']
    #
    # actuals_return_interval = test_df['return_interval']
    # actuals_returns = test_df['return']
    #
    # del test_df['return_interval']
    # del test_df['return']
    # del test_df['timestamp']
    # del test_df['datetime_utc']
    #
    # train_columns = train_df.columns.tolist()
    # test_columns = [col for col in train_columns if col in test_df.columns]
    # test_df = test_df[test_columns]
    #
    # # model = BitcoinTradingModel()
    # #
    # # model.train(data=train_df)
    # #
    # # model.save('/Users/Petru_Buzulan/Private/bi/workspace/abi/models/NOT_BTC_return_interval_prediction_model.pkl')
    #
    # # f_test_data = '/Users/Petru_Buzulan/Private/bi/workspace/abi/data/test/BTC_ML_NORMALIZED_TEST_DATA.csv'
    #
    # # df_test_data = pd.read_csv(f_test_data)
    #
    # # del df_test_data['return']
    # # del df_test_data['timestamp']
    # # del df_test_data['datetime_utc']
    #
    # # metrics = model.calculate_metrics(test_df, actuals_return_interval)
    # # print(metrics)
    # #
    # # print('\n\n')
    # #
    # # prediction = model.predict(test_df)
    # # print(prediction)
    #
    # # df_test_data = pd.read_csv(
    # #     '/Users/Petru_Buzulan/Private/bi/workspace/abi/data/test/1_ROW_BTC_ML_NORMALIZED_TEST_DATA.csv')
    # #
    # # actuals_return_interval = df_test_data['return_interval']
    #
    # # del df_test_data['return']
    # # del df_test_data['timestamp']
    # # del df_test_data['datetime_utc']
    # # del df_test_data['return_interval']
    #
    # model = BitcoinTradingModel.load(
    #     '/Users/Petru_Buzulan/Private/bi/workspace/abi/models/NORMALIZED_BTC_return_interval_prediction_model.pkl')
    # #
    # # prediction = model.predict(test_df)
    # # #
    # metrics = model.calculate_metrics(test_df, actuals_return_interval, actuals_returns)
    #
    # print(metrics)
    # print(prediction)
    #
    # predicted_returns = [pred[1] for pred in prediction]
    # position = [pred[0] for pred in prediction]
    #
    # test_df['return_interval'] = actuals_return_interval
    # test_df['predicted_returns'] = predicted_returns
    # test_df['position'] = position
    #
    # test_df.to_csv('/Users/Petru_Buzulan/Private/bi/workspace/abi/data/analysis/actuals_vs_pred.csv', index=False)
