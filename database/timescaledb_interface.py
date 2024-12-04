import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
from utils.config import Config
from utils.logger import get_logger

class TimescaleDBInterface:
    """
    Class for interacting with TimescaleDB.
    """

    def __init__(self):
        """
        Initialize the database connection.
        """
        self.logger = get_logger(self.__class__.__name__)
        try:
            self.conn = psycopg2.connect(
                dbname=Config.DB_CONFIG['dbname'],
                user=Config.DB_CONFIG['user'],
                password=Config.DB_CONFIG['password'],
                host=Config.DB_CONFIG['host'],
                port=Config.DB_CONFIG['port']
            )
            self.cursor = self.conn.cursor()
            self.logger.info("Connected to TimescaleDB successfully.")
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            raise

    def create_tables(self, feature_columns):
        """
        Create necessary tables in the database.

        :param feature_columns: List of feature column names for the features_data table.
        """
        try:
            # Create OHLC Data Table
            ohlc_table_query = """
            CREATE TABLE IF NOT EXISTS ohlc_data (
                timestamp TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                volume DOUBLE PRECISION,
                timeframe VARCHAR(10),
                PRIMARY KEY (timestamp, timeframe)
            );
            """
            self.cursor.execute(ohlc_table_query)
            self.conn.commit()
            self.logger.info("OHLC data table created successfully.")

            # Convert to Hypertable
            hypertable_query = """
            SELECT create_hypertable('ohlc_data', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE);
            """
            self.cursor.execute(hypertable_query)
            self.conn.commit()

            # Create Features Table
            self.create_features_table(feature_columns)

        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            self.conn.rollback()
            raise

    def create_features_table(self, feature_columns):
        """
        Create the features_data table in the database.

        :param feature_columns: List of feature column names.
        """
        try:
            # Drop the existing table if it exists
            self.cursor.execute("DROP TABLE IF EXISTS features_data;")
            self.conn.commit()
            self.logger.info("Existing features_data table dropped.")


            # Construct feature columns definition
            feature_definitions = ',\n'.join([f"{col} DOUBLE PRECISION" for col in feature_columns if col != 'timestamp'])
            features_table_query = f"""
            CREATE TABLE IF NOT EXISTS features_data (
                timestamp TIMESTAMPTZ PRIMARY KEY,
                {feature_definitions}
            );
            """
            self.cursor.execute(features_table_query)
            self.conn.commit()
            self.logger.info("Features table created successfully.")

            # Convert to Hypertable
            hypertable_query = """
            SELECT create_hypertable('features_data', 'timestamp', if_not_exists => TRUE);
            """
            self.cursor.execute(hypertable_query)
            self.conn.commit()

        except Exception as e:
            self.logger.error(f"Failed to create features table: {e}")
            self.conn.rollback()
            raise

    # Helper function to adapt data types
    @staticmethod
    def adapt_type(value):
        if isinstance(value, (np.datetime64, pd.Timestamp)):
            return value.to_pydatetime()
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, np.generic):
            return value.item()
        else:
            return value

    def insert_ohlc_data(self, df, timeframe):
        """
        Insert OHLC data into the database.

        :param df: DataFrame containing OHLC data.
        :param timeframe: Timeframe of the data.
        """
        try:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['timeframe'] = timeframe

            # Convert data types if necessary
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)

            # Prepare records with adapted data types
            records = [
                tuple(self.adapt_type(value) for value in row)
                for row in df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'timeframe']].itertuples(index=False, name=None)
            ]
            query = """
            INSERT INTO ohlc_data (timestamp, open, high, low, close, volume, timeframe)
            VALUES %s ON CONFLICT (timestamp, timeframe) DO NOTHING;
            """
            execute_values(self.cursor, query, records)
            self.conn.commit()
            self.logger.info(f"Inserted {len(df)} rows into ohlc_data.")
        except Exception as e:
            self.logger.error(f"Failed to insert OHLC data: {e}")
            self.conn.rollback()
            raise

    def insert_features_data(self, df):
        """
        Insert computed features into the database.

        :param df: DataFrame containing computed features.
        """
        try:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Remove 'timeframe' column if it exists
            if 'timeframe' in df.columns:
                df.drop(columns=['timeframe'], inplace=True)
            
            columns = df.columns.tolist()
            records = [
                tuple(self.adapt_type(value) for value in row)
                for row in df[columns].itertuples(index=False, name=None)
            ]
            columns_str = ', '.join(columns)

            update_columns = [f"{col} = EXCLUDED.{col}" for col in columns if col != 'timestamp']
            update_str = ', '.join(update_columns)

            query = f"""
            INSERT INTO features_data ({columns_str})
            VALUES %s
            ON CONFLICT (timestamp) DO UPDATE SET
            {update_str};
            """
            execute_values(self.cursor, query, records)
            self.conn.commit()
            self.logger.info(f"Inserted {len(df)} rows into features_data.")
        except Exception as e:
            self.logger.error(f"Failed to insert features data: {e}")
            self.conn.rollback()
            raise

    def get_ohlc_data(self, timeframe, start_time=None, end_time=None):
        """
        Retrieve OHLC data from the database.

        :param timeframe: Timeframe of the data to retrieve.
        :param start_time: Optional start time for data retrieval.
        :param end_time: Optional end time for data retrieval.
        :return: DataFrame containing OHLC data.
        """
        try:
            query = "SELECT * FROM ohlc_data WHERE timeframe = %s"
            params = [timeframe]
            if start_time:
                query += " AND timestamp >= %s"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= %s"
                params.append(end_time)
            query += " ORDER BY timestamp;"
            df = pd.read_sql_query(query, self.conn, params=params)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            self.logger.info(f"Retrieved {len(df)} rows from ohlc_data.")
            return df
        except Exception as e:
            self.logger.error(f"Failed to retrieve OHLC data: {e}")
            raise

    def get_features_data(self, start_time=None, end_time=None, limit=None):
        """
        Retrieve features data from the database.

        :param start_time: Optional start time for data retrieval.
        :param end_time: Optional end time for data retrieval.
        :param limit: Optional limit for the number of rows to retrieve.
        :return: DataFrame containing features data.
        """
        try:
            query = "SELECT * FROM features_data"
            conditions = []
            params = []

            if start_time:
                conditions.append("timestamp >= %s")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp <= %s")
                params.append(end_time)
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            query += ";"

            df = pd.read_sql_query(query, self.conn, params=params)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df.sort_values('timestamp', inplace=True)  # Ensure chronological order
            self.logger.info(f"Retrieved {len(df)} rows from features_data.")
            return df
        except Exception as e:
            self.logger.error(f"Failed to retrieve features data: {e}")
            raise

    def get_latest_ohlc_data(self, timeframe):
        """
        Retrieve the latest OHLC data point from the database for a given timeframe.

        :param timeframe: Timeframe of the data.
        :return: DataFrame containing the latest OHLC data point.
        """
        try:
            query = "SELECT * FROM ohlc_data WHERE timeframe = %s ORDER BY timestamp DESC LIMIT 1;"
            df = pd.read_sql_query(query, self.conn, params=(timeframe,))
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            self.logger.info("Retrieved latest OHLC data point.")
            return df.iloc[0] if not df.empty else None
        except Exception as e:
            self.logger.error(f"Failed to retrieve latest OHLC data: {e}")
            raise

    def get_latest_features_data(self):
        """
        Retrieve the latest features data point from the database.

        :return: Series containing the latest features data point.
        """
        try:
            query = "SELECT * FROM features_data ORDER BY timestamp DESC LIMIT 1;"
            df = pd.read_sql_query(query, self.conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            self.logger.info("Retrieved latest features data point.")
            return df.iloc[0] if not df.empty else None
        except Exception as e:
            self.logger.error(f"Failed to retrieve latest features data: {e}")
            raise

    def get_historical_features(self, sequence_length):
        """
        Retrieve the last 'sequence_length' data points with features.

        :param sequence_length: Number of past data points to retrieve.
        :return: DataFrame with features.
        """
        try:
            query = f"SELECT * FROM features_data ORDER BY timestamp DESC LIMIT {sequence_length};"
            df = pd.read_sql_query(query, self.conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df.sort_values('timestamp', inplace=True)  # Ensure chronological order
            self.logger.info(f"Retrieved last {sequence_length} rows from features_data.")
            return df
        except Exception as e:
            self.logger.error(f"Failed to retrieve historical features: {e}")
            raise

    def update_features(self, df):
        """
        Update features data in the database.

        :param df: DataFrame containing computed features to update.
        """
        try:
            self.insert_features_data(df)
            self.logger.info("Features data updated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to update features data: {e}")
            raise

    def close_connection(self):
        """
        Close the database connection.
        """
        try:
            self.cursor.close()
            self.conn.close()
            self.logger.info("Database connection closed.")
        except Exception as e:
            self.logger.error(f"Failed to close database connection: {e}")
            raise
