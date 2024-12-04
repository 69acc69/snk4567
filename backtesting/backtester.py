import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import backtrader as bt
import pandas as pd
import numpy as np
from ai_models.ai_model import AIModel
from strategy.trading_strategy import TradingStrategy
from feature_engineering.backtest_feature_engineer import FeatureEngineer
from utils.logger import get_logger
from database.timescaledb_interface import TimescaleDBInterface

class Backtester:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.db = TimescaleDBInterface()  # Initialize database interface

    def run_backtest(self):
        final_value = None  # Initialize final_value
        try:
            self.logger.info("Running backtest...")
            print("Running backtest...")  # For immediate console output
            cerebro = bt.Cerebro()
            data = self.get_backtest_data()
            cerebro.adddata(data)
            cerebro.addstrategy(ModelBacktestingStrategy)
            cerebro.broker.set_cash(100000)
            cerebro.addsizer(bt.sizers.FixedSize, stake=1)
            result = cerebro.run()
            final_value = cerebro.broker.getvalue()
            self.logger.info(f"Final Portfolio Value: {final_value}")
            print(f"Final Portfolio Value: {final_value}")
            # Plot the result (optional)
            # cerebro.plot()
        except Exception as e:
            self.logger.error(f"Error during backtest: {e}")
            print(f"Error during backtest: {e}")
        return final_value

    def get_backtest_data(self):
        # Use test split data
        df = self.get_test_data()
        df['datetime'] = pd.to_datetime(df['timestamp'], utc=True)
        df.set_index('datetime', inplace=True)
        data = bt.feeds.PandasData(
            dataname=df,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            datetime=None,
            openinterest=-1,
            tz='UTC',
        )
        return data

    def get_test_data(self):
        # Fetch the OHLC data
        df = self.db.get_ohlc_data('1m')
        df.sort_values('timestamp', inplace=True)
        # Use the last 10% of data as test data
        test_size = int(0.1 * len(df))
        test_df = df[-test_size:]
        return test_df

class ModelBacktestingStrategy(bt.Strategy):
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing ModelBacktestingStrategy...")
        # Load trained AI models
        self.db = TimescaleDBInterface()
        self.ai_model = AIModel(self.db)
        self.ai_model.load_model()
        # Initialize feature engineer for backtesting
        self.feature_engineer = FeatureEngineer()
        # Initialize trading strategy
        self.trading_strategy = TradingStrategy(self.ai_model, self.feature_engineer)
        # Data feed columns
        self.dataclose = self.datas[0].close
        self.data_datetime = self.datas[0].datetime

    def next(self):
        self.logger.info("Executing next()...")
        # Get the current datetime from Backtrader
        dt = self.datas[0].datetime.datetime(0)
        # Get latest market data
        latest_market_data = pd.DataFrame({
            'timestamp': [dt],
            'open': [self.datas[0].open[0]],
            'high': [self.datas[0].high[0]],
            'low': [self.datas[0].low[0]],
            'close': [self.dataclose[0]],
            'volume': [self.datas[0].volume[0]],
        })
        
        # Generate signal
        signal = self.trading_strategy.generate_signal(latest_market_data)
        self.logger.info(f"Generated signal: {signal}")

        # Implement signal logic
        if signal == 'buy' and not self.position:
            self.buy()
            self.logger.info(f"BUY EXECUTED at {self.dataclose[0]}")
        elif signal == 'sell' and self.position:
            self.sell()
            self.logger.info(f"SELL EXECUTED at {self.dataclose[0]}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.logger.info(f"BUY EXECUTED, Price: {order.executed.price}, Size: {order.executed.size}")
            elif order.issell():
                self.logger.info(f"SELL EXECUTED, Price: {order.executed.price}, Size: {order.executed.size}")
            self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.logger.info(f"OPERATION PROFIT, GROSS {trade.pnl}, NET {trade.pnlcomm}")

if __name__ == '__main__':
    print("Starting backtester...")
    backtester = Backtester()
    backtester.run_backtest()
    print("Backtester finished.")
