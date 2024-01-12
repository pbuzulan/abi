import os
import pandas as pd
import numpy as np

from config.indicators_v1 import INDICATORS_SETTINGS
from src.data_processing.data_cleaning import drop_rows_with_nan_in_columns, remove_sparse_indicators
from src.data_processing.data_transforms import normalize_dataset
from src.indicators_v1.ad_line import calculate_ad_line
from src.indicators_v1.adxr import calculate_adxr
from src.indicators_v1.ama import calculate_ama
from src.indicators_v1.aroon import calculate_aroon
from src.indicators_v1.atr import calculate_atr
from src.indicators_v1.atr_percentage import calculate_atr_percentage
from src.indicators_v1.balance_of_power import calculate_balance_of_power
from src.indicators_v1.bollinger_bands import calculate_bollinger_bands
from src.indicators_v1.cci import calculate_cci
from src.indicators_v1.chaikin_money_flow import calculate_chaikin_money_flow
from src.indicators_v1.donchian_channels import calculate_donchian_channels
from src.indicators_v1.dpo import calculate_dpo
from src.indicators_v1.ehlers_fisher_transform import calculate_ehlers_fisher_transform
from src.indicators_v1.elder_ray_index import calculate_elder_ray_index
from src.indicators_v1.elders_force_index import calculate_elders_force_index
from src.indicators_v1.ema import calculate_ema
from src.indicators_v1.fisher_transform import calculate_fisher_transform
from src.indicators_v1.heikin_ashi import calculate_heikin_ashi
from src.indicators_v1.hla import calculate_hla
from src.indicators_v1.hma import calculate_hma
from src.indicators_v1.ichimoku_cloud import calculate_ichimoku_cloud
from src.indicators_v1.kama import calculate_kama
from src.indicators_v1.keltner_channels import calculate_keltner_channels
from src.indicators_v1.linear_regression_slope import calculate_linear_regression_slope
from src.indicators_v1.macd import calculate_macd
from src.indicators_v1.market_sentiment_oscillator import calculate_market_sentiment_oscillator
from src.indicators_v1.mfi import calculate_mfi
from src.indicators_v1.momentum import calculate_momentum
from src.indicators_v1.obv import calculate_obv
from src.indicators_v1.price_oscillator import calculate_price_oscillator
from src.indicators_v1.roc import calculate_roc
from src.indicators_v1.rsi import calculate_rsi
from src.indicators_v1.sma import calculate_sma
from src.indicators_v1.std_dev import calculate_std_dev
from src.indicators_v1.stochastic_oscillator import calculate_stochastic_oscillator
from src.indicators_v1.tii import calculate_tii
from src.indicators_v1.tsi import calculate_tsi
from src.indicators_v1.ultimate_oscillator import calculate_ultimate_oscillator
from src.indicators_v1.volume_oscillator import calculate_volume_oscillator
from src.indicators_v1.vortex_indicator import calculate_vortex_indicator
from src.indicators_v1.williams_r import calculate_williams_r
from src.indicators_v1.wto import calculate_wto
from src.indicators_v1.zigzag import calculate_zigzag
from src.utils.files_helpers import list_files_in_directory

DATA_RAW_DIR = '../data/raw/'
DATA_INTERMEDIATE_DIR = '../data/intermediate/'
DATA_PROCESSED_DIR = '../data/processed/'


def load_datasets():
    files = list_files_in_directory(DATA_RAW_DIR)
    for f in files:
        df = pd.read_csv(f)
        return df


def apply_indicators_to_dataframe(df: pd.DataFrame, timespan: str = '1h'):
    """
    Apply a series of technical indicators to the given DataFrame.

    :param df: pandas.DataFrame with columns 'open', 'high', 'low', 'close', 'volume'
    :param timespan: str with values '1h', '4h', '1d', '1w'
    :return: pandas.DataFrame with additional columns for each technical indicator
    """
    df = calculate_ad_line(df=df)

    df = calculate_adxr(df=df, timeperiod=INDICATORS_SETTINGS['ADXR']['timeperiod'][timespan])

    df = calculate_ama(df=df, timeperiod=INDICATORS_SETTINGS['AMA']['timeperiod'][timespan])

    df = calculate_aroon(df=df, timeperiod=INDICATORS_SETTINGS['AROON']['timeperiod'][timespan])

    df = calculate_atr(df=df, timeperiod=INDICATORS_SETTINGS['ATR']['timeperiod'][timespan])

    df = calculate_atr_percentage(df=df, timeperiod=INDICATORS_SETTINGS['ATR_PERCENTAGE']['timeperiod'][timespan])

    df = calculate_balance_of_power(df=df)

    df = calculate_bollinger_bands(
        df=df,
        timeperiod=INDICATORS_SETTINGS['BOLLINGER_BANDS']['timeperiod'][timespan],
        nbdevup=INDICATORS_SETTINGS['BOLLINGER_BANDS']['nbdevup'][timespan],
        nbdevdn=INDICATORS_SETTINGS['BOLLINGER_BANDS']['nbdevdn'][timespan]
    )

    df = calculate_cci(df=df, timeperiod=INDICATORS_SETTINGS['CCI']['timeperiod'][timespan])

    df = calculate_chaikin_money_flow(
        df=df,
        fastperiod=INDICATORS_SETTINGS['CMF']['fastperiod'][timespan],
        slowperiod=INDICATORS_SETTINGS['CMF']['slowperiod'][timespan]
    )

    df = calculate_donchian_channels(df=df, timeperiod=INDICATORS_SETTINGS['DONCHIAN_CHANNELS']['timeperiod'][timespan])

    df = calculate_dpo(df=df, timeperiod=INDICATORS_SETTINGS['DPO']['timeperiod'][timespan])

    df = calculate_ehlers_fisher_transform(df=df,
                                           timeperiod=INDICATORS_SETTINGS['EHLERS_FISHER_TRANSFORM']['timeperiod'][
                                               timespan])

    df = calculate_elder_ray_index(df=df, ema_period=INDICATORS_SETTINGS['ELDER_RAY']['ema_period'][timespan])

    df = calculate_elders_force_index(df=df, timeperiod=INDICATORS_SETTINGS['EFI']['timeperiod'][timespan])

    for v_ema in INDICATORS_SETTINGS['EMA']['timeperiods'][timespan]:
        df = calculate_ema(df, v_ema)

    df = calculate_fisher_transform(df=df, lookback=INDICATORS_SETTINGS['FISHER_TRANSFORM']['lookback'][timespan])

    df = calculate_heikin_ashi(df=df)

    df = calculate_hla(df=df, timeperiod=INDICATORS_SETTINGS['HLA']['timeperiod'][timespan])

    df = calculate_hma(df=df, timeperiod=INDICATORS_SETTINGS['HMA']['timeperiod'][timespan])

    df = calculate_ichimoku_cloud(
        df=df,
        conversion_line_period=INDICATORS_SETTINGS['ICHIMOKU_CLOUD']['conversion_line_period'][timespan],
        base_line_period=INDICATORS_SETTINGS['ICHIMOKU_CLOUD']['base_line_period'][timespan],
        lead_span_b_period=INDICATORS_SETTINGS['ICHIMOKU_CLOUD']['lead_span_b_period'][timespan]
    )

    df = calculate_kama(df=df, timeperiod=INDICATORS_SETTINGS['KAMA']['timeperiod'][timespan])

    df = calculate_keltner_channels(
        df=df,
        ema_period=INDICATORS_SETTINGS['KELTNER_CHANNELS']['ema_period'][timespan],
        atr_period=INDICATORS_SETTINGS['KELTNER_CHANNELS']['atr_period'][timespan],
        multiplier=INDICATORS_SETTINGS['KELTNER_CHANNELS']['multiplier'][timespan]
    )

    df = calculate_linear_regression_slope(df=df,
                                           timeperiod=INDICATORS_SETTINGS['LIN_REG_SLOPE']['timeperiod'][timespan])

    df = calculate_macd(
        df=df,
        fastperiod=INDICATORS_SETTINGS['MACD']['fastperiod'][timespan],
        slowperiod=INDICATORS_SETTINGS['MACD']['slowperiod'][timespan],
        signalperiod=INDICATORS_SETTINGS['MACD']['signalperiod'][timespan],
    )

    df = calculate_market_sentiment_oscillator(df=df, timeperiod=
    INDICATORS_SETTINGS['MARKET_SENTIMENT_OSCILLATOR']['timeperiod'][timespan])

    df = calculate_mfi(df=df)

    df = calculate_momentum(df=df, timeperiod=INDICATORS_SETTINGS['MOMENTUM']['timeperiod'][timespan])

    df = calculate_obv(df=df)

    df = calculate_price_oscillator(
        df=df,
        short_period=INDICATORS_SETTINGS['PRICE_OSC']['short'][timespan],
        long_period=INDICATORS_SETTINGS['PRICE_OSC']['long'][timespan]
    )

    df = calculate_roc(df=df, timeperiod=INDICATORS_SETTINGS['ROC']['timeperiod'][timespan])

    df = calculate_rsi(df=df, timeperiod=INDICATORS_SETTINGS['RSI']['timeperiod'][timespan])

    for v_sma in INDICATORS_SETTINGS['SMA']['timeperiods'][timespan]:
        df = calculate_sma(df, v_sma)

    df = calculate_std_dev(df=df, timeperiod=INDICATORS_SETTINGS['STD_DEV']['timeperiod'][timespan])

    df = calculate_stochastic_oscillator(
        df=df,
        fastk_period=INDICATORS_SETTINGS['STOCH']['fastk_period'][timespan],
        slowk_period=INDICATORS_SETTINGS['STOCH']['slowk_period'][timespan],
        slowd_period=INDICATORS_SETTINGS['STOCH']['slowd_period'][timespan]
    )

    df = calculate_tii(df=df, timeperiod=INDICATORS_SETTINGS['TII']['timeperiod'][timespan])

    df = calculate_tsi(
        df=df,
        high_period=INDICATORS_SETTINGS['TSI']['high_period'][timespan],
        low_period=INDICATORS_SETTINGS['TSI']['low_period'][timespan]
    )

    df = calculate_ultimate_oscillator(
        df=df,
        timeperiod1=INDICATORS_SETTINGS['ULT_OSC']['timeperiod1'][timespan],
        timeperiod2=INDICATORS_SETTINGS['ULT_OSC']['timeperiod2'][timespan],
        timeperiod3=INDICATORS_SETTINGS['ULT_OSC']['timeperiod3'][timespan]
    )

    df = calculate_volume_oscillator(
        df=df,
        short_period=INDICATORS_SETTINGS['VOL_OSC']['short'][timespan],
        long_period=INDICATORS_SETTINGS['VOL_OSC']['long'][timespan]
    )

    df = calculate_vortex_indicator(df=df, timeperiod=INDICATORS_SETTINGS['VORTEX']['timeperiod'][timespan])

    df = calculate_williams_r(df=df, timeperiod=INDICATORS_SETTINGS['WILLIAMS_R']['timeperiod'][timespan])

    df = calculate_wto(
        df=df,
        n1=INDICATORS_SETTINGS['WTO']['n1'][timespan],
        n2=INDICATORS_SETTINGS['WTO']['n2'][timespan]
    )

    # df = calculate_zigzag(df=df, threshold_percentage=INDICATORS_SETTINGS['ZIGZAG']['threshold'][timespan])

    return df


if __name__ == '__main__':
    df = load_datasets()
    df = apply_indicators_to_dataframe(df)

    df = drop_rows_with_nan_in_columns(df)
    df = remove_sparse_indicators(df)

    df = normalize_dataset(df)

    df.to_csv(f'{DATA_PROCESSED_DIR}/BTC.csv', index=False)
