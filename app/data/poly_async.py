import asyncio
import pandas as pd
import polygon


async def fetch_ohlc_data(api_key, symbol, from_date, to_date):
    # Initialize the Polygon client
    async with polygon.CryptoClient(api_key, True) as client:
        # Fetch the aggregate bars
        bars = await client.get_aggregate_bars(symbol, from_date, to_date, timespan='hour', full_range=True)

        # Assuming 'bars' is an iterable of bar objects. Adjust attribute names as per the actual object structure.
        data = [{
            'timestamp': bar['t'],
            'open': bar['o'],
            'high': bar['h'],
            'low': bar['l'],
            'close': bar['c'],
            'volume': bar['v']
        } for bar in bars]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Convert Timestamps to readable format (if needed)
        df['datetime_utc'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Save to CSV
        df.to_csv(f'../../resources/polygon/data_1h/{symbol[2:].split("USD")[0]}_ohlc_1h.csv', index=False)


# TODO: initialize symbols from resources/polygon/tickers_simplified.json

for symbol in symbols:
    print(f'fetching coin {symbol} now...')
    # Replace with your actual API key, symbol, and date range
    api_key = '_4VGTK0ot6gDBrL1ppt9KVbONy1m8MAT'
    # symbol = 'X:ETHUSD'
    from_date = '2013-06-1'
    to_date = '2024-01-7'

    # Run the async function
    asyncio.run(fetch_ohlc_data(api_key, symbol, from_date, to_date))

    print(f"Completed processing item {symbol} \n\n")
