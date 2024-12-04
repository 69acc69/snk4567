import sys
import argparse
from datetime import datetime, timedelta, timezone
from utils.config import Config
from utils.logger import get_logger
from database.timescaledb_interface import TimescaleDBInterface
from data_collection.ohlc_data import OHLCDataCollector
from feature_engineering.feature_engineer import FeatureEngineer
from ai_models.ai_model import AIModel
from strategy.trading_strategy import TradingStrategy
from execution.trade_executor import TradeExecutor

def main():
    logger = get_logger('Main')
    logger.info("Starting Trading Bot")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--collect-data', action='store_true', help='Collect data from APIs')
    parser.add_argument('--compute-features', action='store_true', help='Compute features')
    parser.add_argument('--train-models', action='store_true', help='Train AI models')
    parser.add_argument('--run-strategy', action='store_true', help='Run trading strategy')
    parser.add_argument('--initialize-db', action='store_true', help='Initialize database tables')
    parser.add_argument('--symbol', type=str, default='ETHUSD', help='Trading symbol')  # Adjusted to match ccxt format
    parser.add_argument('--timeframe', type=str, default='1m', help='Timeframe for OHLC data')
    parser.add_argument('--data-limit', type=int, default=1000, help='Number of data points to fetch')
    args = parser.parse_args()

    # Initialize database interface
    db_interface = TimescaleDBInterface()

    # Initialize database tables if needed
    if args.initialize_db:
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'vsa_category', 'bos', 'liquidity_zone',
            'vwap_30m', 'vwap_daily', 'ma_20', 'ma_50',
            'atr_14', 'engulfing', 'higher_high', 'lower_low',
            'pullback_to_vwap_30m', 'pullback_to_vwap_daily',
            'vsa_category_5min', 'bos_5min', 'liquidity_zone_5min',
            'vsa_category_15min', 'bos_15min', 'liquidity_zone_15min',
            'vsa_category_1H', 'bos_1H', 'liquidity_zone_1H'
        ]
        try:
            db_interface.create_tables(feature_columns)
            logger.info("Database tables initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
            sys.exit(1)

    # Data Collection
    if args.collect_data:
        logger.info("Collecting data...")
        try:
            # OHLC Data Collection
            logger.info("Starting OHLC data collection...")
            ohlc_collector = OHLCDataCollector(Config.API_KEY, Config.API_SECRET)
            since_date = datetime.now(timezone.utc) - timedelta(days=730)  # Approximate 2 years
            since_timestamp = int(since_date.timestamp() * 1000)  # Convert to milliseconds
            ohlc_df = ohlc_collector.fetch_historical_ohlc(args.symbol, args.timeframe, since_timestamp=since_timestamp)
            if not ohlc_df.empty:
                db_interface.insert_ohlc_data(ohlc_df, args.timeframe)
                logger.info("OHLC data inserted into the database.")
            else:
                logger.warning("No OHLC data fetched.")

            logger.info("Data collection completed successfully.")
        except Exception as e:
            logger.error(f"Error during data collection: {e}", exc_info=True)
            sys.exit(1)

    # Feature Engineering
    if args.compute_features:
        logger.info("Computing features...")
        try:
            feature_engineer = FeatureEngineer(db_interface)
            feature_engineer.compute_features()
            logger.info("Feature engineering completed successfully.")
        except Exception as e:
            logger.error(f"Error during feature engineering: {e}")
            sys.exit(1)

    # AI Model Training
    if args.train_models:
        logger.info("Training AI models...")
        try:
            ai_model = AIModel(db_interface)
            # ai_model.train_lstm()
            # ai_model.save_model(model_name='lstm')

            # ai_model.train_xgboost_with_scaling()
            # ai_model.save_model(model_name='xgboost')

            ai_model.train_ppo_agent()
            ai_model.save_model(model_name='ppo')

            logger.info("AI models trained and saved successfully.")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            sys.exit(1)

    # Strategy Execution
    if args.run_strategy:
        logger.info("Running trading strategy...")
        try:
            # Initialize TradeExecutor with API keys and symbol
            trade_executor = TradeExecutor(Config.API_KEY, Config.API_SECRET, args.symbol)
            # Execute a trade cycle
            trade_executor.execute_trade_cycle()
            logger.info("Trading strategy executed successfully.")
        except Exception as e:
            logger.error(f"Error during strategy execution: {e}")
            sys.exit(1)

    logger.info("Trading Bot operations completed")

if __name__ == '__main__':
    main()
