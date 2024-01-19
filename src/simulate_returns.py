import os
import pandas as pd
from src.analysis.data_analysis import simulate_return

DATA_RAW_DIR = os.getenv('DATA_RAW_DIR', '../data/raw/')
DATA_INTERMEDIATE_DIR = os.getenv('DATA_INTERMEDIATE_DIR', '../data/intermediate/')
DATA_PROCESSED_DIR = os.getenv('DATA_PROCESSED_DIR', '../data/processed/')
DATA_TEST_DIR = os.getenv('DATA_TEST_DIR', '../data/test/')
DATA_TRAINING_DIR = os.getenv('DATA_TRAINING_DIR', '../data/training/')
DATA_ANALYSIS_DIR = os.getenv('DATA_ANALYSIS_DIR', '../data/analysis/')
MODELS_DIR = os.getenv('MODELS_DIR', '../models/')
ANALYSIS_FILE_NAME = os.getenv('ANALYSIS_FILE_NAME', '../data/analysis/1D_BTC_Analysis_Data_BitcoinTradingModel.csv')

if __name__ == '__main__':
    # SIMULATE CASH RETURN
    _df = pd.read_csv(ANALYSIS_FILE_NAME)
    return_cash = simulate_return(_df, budget=1000, leverage=10)
    # print(return_cash)
