import requests
import polygon


def fetch_all_crypto_tickers():
    base_url = 'https://api.polygon.io/v3/reference/tickers'
    api_key = 'K4o7ZdJawXVJcnsx3m05tLsiOWKuYvrz'
    params = {
        'market': 'crypto',
        'active': 'true',
        'apiKey': api_key
    }

    all_tickers = []
    next_url = base_url

    while next_url:
        response = requests.get(next_url, params=params)
        if response.status_code == 200:
            data = response.json()
            all_tickers.extend(data['results'])

            # Check if there is a next page
            next_url = data.get('next_url')
            # params = {}  # Clear params since next_url includes necessary parameters
        else:
            print("Failed to fetch data:", response.status_code)
            break

    return all_tickers


def fetch_all_crypto_news():
    base_url = 'https://api.polygon.io/v2/reference/news?'
    api_key = '_4VGTK0ot6gDBrL1ppt9KVbONy1m8MAT'
    params = {
        'ticker': 'X:BTCUSD',
        'apiKey': api_key
    }

    all_tickers = []
    next_url = base_url

    response = requests.get(next_url, params=params)
    if response.status_code == 200:
        data = response.json()
        all_tickers.extend(data['results'])

        # Check if there is a next page
        next_url = data.get('next_url')
        # params = {}  # Clear params since next_url includes necessary parameters
    else:
        print("Failed to fetch data:", response.status_code)

    return all_tickers


fetch_all_crypto_news()
