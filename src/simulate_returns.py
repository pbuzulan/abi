import pandas as pd
from src.analysis.data_analysis import simulate_return

if __name__ == '__main__':
    # SIMULATE CASH RETURN
    _df = pd.read_csv('../data/analysis/actuals_vs_pred_v8.csv')
    return_cash = simulate_return(_df)
