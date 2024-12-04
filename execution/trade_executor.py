# trade_executor.py

import ccxt
from utils.logger import get_logger
from utils.config import Config
from strategy.trading_strategy import TradingStrategy
from ai_models.ai_model import AIModel
from feature_engineering.feature_engineer import FeatureEngineer
from database.timescaledb_interface import TimescaleDBInterface
import pandas as pd

class TradeExecutor:
    def __init__(self, api_key, secret, symbol):
        self.exchange = ccxt.bybit({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
        })
        self.exchange.load_markets()
        self.symbol = symbol
        self.logger = get_logger(self.__class__.__name__)

        # Initialize database interface
        self.db_interface = TimescaleDBInterface()

        # Initialize AI Model and load models
        self.ai_model = AIModel(self.db_interface)
        self.ai_model.load_model()

        # Initialize Feature Engineer
        self.feature_engineer = FeatureEngineer(self.db_interface)

        # Initialize Trading Strategy
        self.trading_strategy = TradingStrategy(ai_model=self.ai_model, feature_engineer=self.feature_engineer)

    def execute_trade_cycle(self):
        """
        Execute a single trade cycle: fetch data, generate signal, and execute trade.
        """
        # Fetch latest market data
        latest_market_data = self.fetch_latest_market_data()

        if latest_market_data is None:
            self.logger.error("Failed to fetch latest market data.")
            return

        # Generate trading signal
        signal = self.trading_strategy.generate_signal(latest_market_data)

        # Execute trade based on signal
        self.execute_trade(signal, latest_market_data)

    def fetch_latest_market_data(self):
        """
        Fetch the latest market data for the symbol.
        """
        try:
            # Fetch latest OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=Config.TIMEFRAME, limit=1)
            if not ohlcv:
                self.logger.error("No OHLCV data fetched.")
                return None

            # Extract data
            timestamp, open_price, high_price, low_price, close_price, volume = ohlcv[0]
            latest_market_data = pd.Series({
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })

            return latest_market_data

        except Exception as e:
            self.logger.error(f"Error fetching latest market data: {e}")
            return None

    def execute_trade(self, signal, latest_market_data):
        """
        Execute trade based on the generated signal.
        """
        if signal in ['buy', 'sell']:
            # Compute features
            features = self.feature_engineer.compute_latest_features(latest_market_data)

            # Calculate position size
            position_size = self.calculate_position_size(features, Config.RISK_PER_TRADE)
            if position_size <= 0:
                self.logger.error("Calculated position size is zero or negative.")
                return

            # Calculate stop loss and take profit
            stop_loss, take_profit = self.calculate_stop_loss_take_profit(features, signal)
            if stop_loss is None or take_profit is None:
                self.logger.error("Failed to calculate stop loss or take profit levels.")
                return

            # Place order
            self.place_order(self.symbol, signal, position_size, stop_loss, take_profit)
        else:
            self.logger.info(f"No trade executed for signal: {signal}")

    def calculate_stop_loss_take_profit(self, features, signal):
        """
        Calculate dynamic stop loss and take profit levels.
        """
        atr = features.get('atr_15m') or features.get('atr_14')
        close_price = features.get('close')
        if atr is None or close_price is None:
            self.logger.error("Missing ATR or close price in features.")
            return None, None
        if signal == 'buy':
            stop_loss = close_price - atr
            take_profit = close_price + (atr * 2)
        elif signal == 'sell':
            stop_loss = close_price + atr
            take_profit = close_price - (atr * 2)
        else:
            return None, None
        return stop_loss, take_profit

    def calculate_position_size(self, features, risk_per_trade):
        """
        Calculate position size based on risk management.
        """
        account_balance = self.get_account_balance()
        if account_balance <= 0:
            self.logger.error("Account balance is insufficient.")
            return 0
        atr = features.get('atr_15m') or features.get('atr_14')
        if atr is None or atr <= 0:
            self.logger.error("Invalid ATR value.")
            return 0
        position_size = (account_balance * risk_per_trade) / atr
        # Get market info
        market = self.exchange.markets.get(self.symbol)
        if not market:
            self.logger.error(f"Market data for {self.symbol} not available.")
            return 0
        min_amount = market['limits']['amount']['min'] or 0
        step_size = market['precision']['amount'] or 1e-8  # Use a small default if not provided
        position_size = max(position_size, min_amount)
        # Adjust position size to meet exchange's step size
        position_size = round(position_size / step_size) * step_size
        return position_size

    def get_account_balance(self):
        """
        Get the account balance.
        """
        try:
            balance = self.exchange.fetch_balance()
            currency = Config.CURRENCY
            total_balance = balance['free'].get(currency)
            if total_balance is None:
                self.logger.error(f"Could not retrieve balance for currency {currency}")
                return 0
            return total_balance
        except Exception as e:
            self.logger.error(f"Error fetching account balance: {e}")
            return 0

    def place_order(self, symbol, signal, position_size, stop_loss, take_profit):
        """
        Place an order on the exchange.
        """
        try:
            side = 'buy' if signal == 'buy' else 'sell'
            # Place market order
            order = self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=position_size
            )
            self.logger.info(f"Market order placed: {order}")
            # Place stop-loss and take-profit orders
            if self.exchange.id == 'bybit':
                # For Bybit, use conditional orders
                if signal == 'buy':
                    stop_loss_side = 'sell'
                    take_profit_side = 'sell'
                else:
                    stop_loss_side = 'buy'
                    take_profit_side = 'buy'
                # Stop-Loss Order
                stop_loss_order = self.exchange.create_order(
                    symbol=symbol,
                    type='stop_market',
                    side=stop_loss_side,
                    amount=position_size,
                    params={
                        'stopPrice': stop_loss,
                        'triggerPrice': stop_loss,
                    }
                )
                self.logger.info(f"Stop-loss order placed: {stop_loss_order}")
                # Take-Profit Order
                take_profit_order = self.exchange.create_order(
                    symbol=symbol,
                    type='take_profit_market',
                    side=take_profit_side,
                    amount=position_size,
                    params={
                        'stopPrice': take_profit,
                        'triggerPrice': take_profit,
                    }
                )
                self.logger.info(f"Take-profit order placed: {take_profit_order}")
            else:
                self.logger.warning("Conditional orders not implemented for this exchange.")
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            # Optionally, implement retry logic or alerting mechanisms
