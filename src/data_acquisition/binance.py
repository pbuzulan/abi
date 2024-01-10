import pandas as pd
from datetime import datetime, timezone, timedelta
import calendar
import time
import json
import requests
import concurrent.futures


def get_tickers():
    # Construct the API URL
    url = f"https://api.binance.com/api/v3/exchangeInfo"

    symbols = []
    # Make the API request
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for symbol in data['symbols']:
            if symbol['quoteAsset'] == 'USDT' and symbol['status'] == 'TRADING':
                symbols.append(symbol['symbol'])

        with open('../resources/binance_symbols.json', 'w') as f:
            f.write(json.dumps(symbols))


def get_klines_iter(symbol, interval, start, end=None, limit=1000):
    # start and end must be isoformat YYYY-MM-DD
    # We are using utc time zone

    # the maximum records is 1000 per each Binance API call

    df = pd.DataFrame()

    if start is None:
        print('start time must not be None')
        return
    start = calendar.timegm(datetime.fromisoformat(start).timetuple()) * 1000

    if end is None:
        dt = datetime.now(timezone.utc)
        utc_time = dt.replace(tzinfo=timezone.utc)
        end = int(utc_time.timestamp()) * 1000
        return
    else:
        end = calendar.timegm(datetime.fromisoformat(end).timetuple()) * 1000
    last_time = None

    while len(df) == 0 or (last_time is not None and last_time < end):
        url = 'https://api.binance.com/api/v3/klines?symbol=' + \
              symbol + '&interval=' + interval + '&limit=1000'
        if (len(df) == 0):
            url += '&startTime=' + str(start)
        else:
            url += '&startTime=' + str(last_time)

        url += '&endTime=' + str(end)
        df2 = pd.read_json(url)
        df2.columns = ['Opentime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Closetime',
                       'Quote asset volume', 'Number of trades', 'Taker by base', 'Taker buy quote', 'Ignore']
        dftmp = pd.DataFrame()
        dftmp = pd.concat([df2, dftmp], axis=0, ignore_index=True, keys=None)

        dftmp.Opentime = pd.to_datetime(dftmp.Opentime, unit='ms')
        dftmp['Date'] = dftmp.Opentime.dt.strftime("%d/%m/%Y")
        dftmp['Time'] = dftmp.Opentime.dt.strftime("%H:%M:%S")
        dftmp = dftmp.drop(['Quote asset volume', 'Closetime', 'Opentime',
                            'Number of trades', 'Taker by base', 'Taker buy quote', 'Ignore'], axis=1)
        column_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
        dftmp.reset_index(drop=True, inplace=True)
        dftmp = dftmp.reindex(columns=column_names)
        string_dt = str(dftmp['Date'][len(dftmp) - 1]) + 'T' + str(dftmp['Time'][len(dftmp) - 1]) + '.000Z'
        utc_last_time = datetime.strptime(string_dt, "%d/%m/%YT%H:%M:%S.%fZ")
        last_time = (utc_last_time - datetime(1970, 1, 1)) // timedelta(milliseconds=1)
        df = pd.concat([df, dftmp], axis=0, ignore_index=True, keys=None)
    df.to_csv(f'../../resources/binance/{symbol}_{interval}.csv', index=False)


# for coin in coins:
#     start_time = time.time()
#
#     print(f'fetching coin {coin} now...')
#     get_klines_iter(coin, '1h', '2016-01-01', '2024-01-06')
#
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#
#     print(f'done... The method took {elapsed_time} seconds to complete. \n\n')

for coin in coins:
    print(f'fetching coin {coin} now...')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(get_klines_iter, coin, '1h', '2016-01-01', '2024-01-06')
        try:
            # Wait for 180 seconds for your_method to complete
            result = future.result(timeout=180)
        except concurrent.futures.TimeoutError:
            print(f"Method took longer than 180 seconds for item {coin}, moving to next item.")
            continue  # Skip to next item
        # Rest of your code for processing the item
        print(f"Completed processing item {coin}")
