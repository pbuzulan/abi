# # # # from config.logs import setup_logging
# # # #
# # # # setup_logging()
# # #
# # import pandas as pd
# #
# # df = pd.read_csv('/Users/Petru_Buzulan/Private/bi/workspace/abi/data/analysis/actuals_vs_pred.csv')
# #
# #
# # def determine_position(row):
# #     if row['return_interval'] in [4, 5, 6, 7, 9, 10]:
# #         return 'Long'
# #     elif row['return_interval'] in [-8, -7]:
# #         return 'Short'
# #     else:
# #         return 'Cash'
# #
# #
# # # Apply the function to each row of the DataFrame
# # df['actual_position'] = df.apply(determine_position, axis=1)
# #
# # count_same_values = (df['actual_position'] == df['predicted_position'])
# #
# # df['Percentage Change'] = ((df['close.1'] - df['open.1']) / df['open.1']) * 100
# #
# # df.to_csv('/Users/Petru_Buzulan/Private/bi/workspace/abi/data/analysis/actuals_vs_pred_v2.csv', index=False)
# # #
# # # Assuming df is your DataFrame with columns: 'predicted_position', 'Percentage Change'
# # # Starting budget and leverage settings
# # budget = 1000
# # leverage = 10
# #
# # # Initial total value (budget with leverage)
# # total_value = budget * leverage
# #
# # for index, row in df.iterrows():
# #     trade_earning = 0
# #     if row['predicted_position'] == 'Long':
# #         # If long and the percentage change is >= 1%, we gain
# #         trade_earning = total_value * row['Percentage Change'] * leverage
# #
# #     elif row['predicted_position'] == 'Short':
# #         # If short and the percentage change is <= -1%, we gain
# #         trade_earning = total_value * -row['Percentage Change'] * leverage
# #
# #     # Update the total value based on the trade earning
# #     total_value += trade_earning
# #
# #     # Closing the position after each trade
# #     if trade_earning != 0:
# #         print(f"Trade at index {index} closed with earning: {trade_earning}")
# #         break  # Remove this break if you want to continue trading after each gain
# #
# # final_budget = total_value / leverage
# # print(f"Final budget after the simulation: {final_budget}")
#
#
# maint_lookup_table = [
#     (50_000, 0.4, 0),
#     (250_000, 0.5, 50),
#     (1_000_000, 1.0, 1_300),
#     (10_000_000, 2.5, 16_300),
#     (20_000_000, 5.0, 266_300),
#     (50_000_000, 10.0, 1_266_300),
#     (100_000_000, 12.5, 2_516_300),
#     (200_000_000, 15.0, 5_016_300),
#     (300_000_000, 25.0, 25_016_300),
#     (500_000_000, 50.0, 100_016_300),
# ]
#
#
# def binance_btc_liq_balance(wallet_balance, contract_qty, entry_price):
#     for max_position, maint_margin_rate_pct, maint_amount in maint_lookup_table:
#         maint_margin_rate = maint_margin_rate_pct / 100
#         liq_price = (wallet_balance + maint_amount - contract_qty * entry_price) / (
#                 abs(contract_qty) * (maint_margin_rate - (1 if contract_qty >= 0 else -1)))
#         base_balance = liq_price * abs(contract_qty)
#         if base_balance <= max_position:
#             break
#     return round(liq_price, 2)
#
#
# def binance_btc_liq_leverage(leverage, contract_qty, entry_price):
#     wallet_balance = abs(contract_qty) * entry_price / leverage
#     print('[Wallet-balance-equivalent of %s] ' % wallet_balance, end='')
#     return binance_btc_liq_balance(wallet_balance, contract_qty, entry_price)
#
#
# print(binance_btc_liq_balance(1000, -10000, 41983))
