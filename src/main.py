# import pandas as pd
#
# df = pd.read_csv('/Users/Petru_Buzulan/Private/bi/workspace/abi/data/processed/BTC_ML_NORMALIZED.csv')
#
# split_timestamp = 1672531200000
# train_df = df[df['timestamp'] < split_timestamp]
# test_df = df[df['timestamp'] >= split_timestamp]
#
# del df['return']
# del df['timestamp']
# del df['datetime_utc']
#
# # Saving the datasets
# train_df.to_csv('/Users/Petru_Buzulan/Private/bi/workspace/abi/data/test/BTC_ML_NORMALIZED_TRAINING_DATA.csv',
#                 index=False)
# test_df.to_csv('/Users/Petru_Buzulan/Private/bi/workspace/abi/data/test/BTC_ML_NORMALIZED_TEST_DATA.csv', index=False)
#
# return_interval = test_df['return_interval']
# del test_df['return_interval']
#
# # metrics = model.calculate_metrics(test_df, return_interval)
# #
# # print(metrics)
#
# # df_test_data = pd.read_csv(
# #     '/Users/Petru_Buzulan/Private/bi/workspace/abi/data/test/1_ROW_BTC_ML_NORMALIZED_TEST_DATA.csv')
# #
# # del df_test_data['return']
# # del df_test_data['timestamp']
# # del df_test_data['datetime_utc']
# #
# # model = BitcoinTradingModel()
# # model.load('/Users/Petru_Buzulan/Private/bi/workspace/abi/models/BTC_return_interval_prediction_model.pkl')
# #
# # metrics = model.calculate_metrics(df_test_data, df_test_data['return_interval'])
# #
# # prediction = model.predict(df_test_data)
# # print(metrics)
# # print(prediction)
