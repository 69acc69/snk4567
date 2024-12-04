import asyncio
import json
import ccxt
import pandas as pd
import time
import websockets
from database.timescaledb_interface import TimescaleDBInterface
from utils.logger import get_logger
from datetime import datetime, timedelta

class OHLCDataCollector:
    """
    Class for collecting OHLC data from Bybit and storing it into TimescaleDB.
    """
    def __init__(self, api_key, secret):
        """
        Initialize the OHLCDataCollector with Bybit API credentials.
        """
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'adjustForTimeDifference': True,
            'options': {
                'defaultType': 'inverse',
                'recvWindow': 10000,
            }            
        })
        self.exchange.load_time_difference()
        self.exchange.load_markets()
        self.db = TimescaleDBInterface()
        self.logger = get_logger(self.__class__.__name__)

    def fetch_historical_ohlc(self, symbol, timeframe, since_timestamp=None):
        """
        Fetch historical OHLC data starting from 'since_timestamp' up to the current time.

        :param symbol: Trading pair symbol, e.g., 'ETHUSD'.
        :param timeframe: Timeframe string, e.g., '1m'.
        :param since_timestamp: Unix timestamp in milliseconds to start fetching data from.
        """
        all_data = []
        request_count = 0
        since = since_timestamp  # Start timestamp
        max_limit_per_request = 200  # Bybit's maximum limit per request

        while True:
            try:
                data = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=max_limit_per_request)
                if not data:
                    break
                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                all_data.append(df)
                since = data[-1][0] + 1  # Update 'since' to the last timestamp + 1ms
                request_count += 1
                self.logger.info(f"Fetched {len(data)} rows for {timeframe} in request #{request_count}")
                time.sleep(self.exchange.rateLimit / 1000)  # Respect rate limits
            except ccxt.BaseError as e:
                self.logger.error(f"An error occurred: {e}", exc_info=True)
                break

        if all_data:
            all_df = pd.concat(all_data, ignore_index=True)
            return all_df
        else:
            self.logger.warning(f"No data fetched for timeframe {timeframe}.")
            return pd.DataFrame()

    async def collect_live_ohlc(self, symbol, timeframes):
        """
        Collect live OHLC data using WebSocket.

        :param symbol: Trading pair symbol, e.g., 'ETHUSD'.
        :param timeframes: List of timeframes, e.g., ['1m', '15m'].
        """
        async with websockets.connect('wss://stream.bybit.com/v5/public/inverse') as websocket:
            for timeframe in timeframes:
                await websocket.send(json.dumps({
                    "op": "subscribe",
                    "args": [f"candle.{timeframe}.{symbol}"]
                }))
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    if 'topic' in data and 'data' in data:
                        df = pd.DataFrame(data['data'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        self.db.insert_ohlc_data(df, timeframe)
                        self.logger.info(f"Inserted live OHLC data for {timeframe}")
                except Exception as e:
                    self.logger.error(f"WebSocket error: {e}")
                    await asyncio.sleep(5)