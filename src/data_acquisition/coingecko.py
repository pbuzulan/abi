import requests
import pytz
import json
import time
import pandas as pd
from datetime import datetime, timedelta


def write_coingecko_coins_fetch_status(data: dict):
    with open('../../data/coingecko_coins_fetch_status.json', 'w') as f:
        return f.write(json.dumps(data))


def get_coingecko_coins_fetch_status() -> dict:
    with open('../../data/coingecko_coins_fetch_status.json', 'r') as f:
        return json.loads(f.read())


def write_binance_coin_conf(data: dict):
    with open('../../data/coingecko_coin_conf.json', 'w') as f:
        return f.write(json.dumps(data))


def get_binance_coin_by_coind_id(coin_id) -> str:
    with open('../../data/binance_coins.json', 'r') as f:
        return json.loads(f.read())[coin_id]


def read_binance_coin_conf() -> dict:
    with open('../../data/coingecko_coin_conf.json', 'r') as f:
        return json.loads(f.read())


def round_timestamp_to_nearest_hour(timestamp):
    # Convert timestamp to datetime in UTC, round to nearest hour, and convert back to timestamp
    dt = datetime.fromtimestamp(timestamp, pytz.utc)
    rounded_dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=dt.minute // 30)
    return int(rounded_dt.timestamp()), rounded_dt


def fetch_data(coin_id: str, start_year=2018, vs_currency='usd'):
    data = []
    last_date_coins = read_binance_coin_conf()
    last_date = last_date_coins.get(coin_id)
    end_date = datetime.now(pytz.utc)
    error_to_raise = False

    if last_date is None:
        start_date = datetime(start_year, 1, 1, 0, 0, 0, 0, tzinfo=pytz.utc)
        # end_date = datetime(start_year, 5, 3, 0, 0, 0, 0, tzinfo=pytz.utc)
    else:
        start_date = datetime.fromisoformat(last_date)

    _next_date = start_date

    while start_date < end_date:
        # CoinGecko free API might not allow fetching hourly data for a long range in one request
        # Adjust the range as per the API limitations
        next_date = start_date + timedelta(days=90)  # Example: 90 days range
        if next_date > end_date:
            next_date = end_date

        print(f"Fetching data from {start_date} to {next_date} for {coin_id}")
        _next_date = next_date

        # Construct the API URL
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency={vs_currency}&from={int(start_date.timestamp())}&to={int(next_date.timestamp())}"

        # Make the API request
        response = requests.get(url)
        if response.status_code == 200:
            historical = response.json()

            # Extract prices and timestamps
            prices = historical.get('prices', [])
            market_caps = historical.get('market_caps', {})
            market_caps_dict = {int(time): cap for time, cap in market_caps}

            for price in prices:
                timestamp, btc_price = price
                btc_market_cap = market_caps_dict.get(int(timestamp), None)
                rounded_timestamp, rounded_datetime = round_timestamp_to_nearest_hour(timestamp / 1000)
                data.append([rounded_timestamp, rounded_datetime, btc_price, btc_market_cap])
        else:
            print(f"Failed to fetch data: {response.status_code}")
            error_to_raise = True
            break

        # Move to the next range
        start_date = next_date

    # Convert to DataFrame
    # df = pd.DataFrame(data, columns=['timestamp', 'price'])
    # return df
    df = pd.DataFrame(data, columns=['unix', 'datetime_utc', 'price', 'market_cap'])
    df['pair'] = f'{get_binance_coin_by_coind_id(coin_id).upper()}/{vs_currency.upper()}'
    try:
        last_datetime = df['datetime_utc'].iloc[-1]
    except Exception:
        last_datetime = _next_date

    last_date_coins[coin_id] = str(last_datetime)
    write_binance_coin_conf(last_date_coins)

    _ = df['datetime_utc'].iloc[-1]

    try:
        existing_df = pd.read_csv(f'../../resources/{coin_id}_60.csv',
                                  names=['unix', 'datetime_utc', 'price', 'market_cap', 'pair'])
    except Exception:
        existing_df = pd.DataFrame()

    combined_df = pd.concat([existing_df, df], ignore_index=True)

    combined_df.to_csv(f'../../resources/{coin_id}_60.csv', index=False)

    if error_to_raise:
        raise Exception()

    return True


def manager(coins_done: list, coins_tbd: list):
    _coins_tbd = coins_tbd
    for coin in coins_tbd:
        try:
            fetch_data(coin_id=coin)
        except Exception:
            print("relaxing for 60 seconds...")
            time.sleep(65)
            manager(coins_done, _coins_tbd)

        coins_done.append(coin)
        _coins_tbd.remove(coin)
        write_coingecko_coins_fetch_status({"done": coins_done, "tbd": _coins_tbd})

        print(f"Coin {coin} completed, file saved.")
        manager(coins_done, _coins_tbd)


if __name__ == '__main__':
    coins_fetch_conf_status = get_coingecko_coins_fetch_status()

    coins_done = coins_fetch_conf_status['done']
    coins_tbd = coins_fetch_conf_status['tbd']
    manager(coins_done, coins_tbd)
