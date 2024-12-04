import pandas as pd
import numpy as np
import talib as ta
from utils.logger import get_logger

class FeatureEngineer:
    """
    Class for computing and managing features used by the AI model.
    """

    def __init__(self, db_interface=None):
        """
        Initialize with a database interface (optional).
        """
        self.db = db_interface
        self.logger = get_logger(self.__class__.__name__)

    def compute_features(self, df=None):
        """
        Compute features from a DataFrame or from the database.

        Parameters:
        - df: DataFrame containing historical data. If None, data will be fetched from the database.

        Returns:
        - df_with_features: DataFrame with computed features.
        """
        if df is None:
            # Retrieve data from the database
            if self.db is None:
                self.logger.error("Database interface not provided. Cannot fetch data.")
                return None
            ohlc_df = self.db.get_ohlc_data('1m')
            ohlc_df['timestamp'] = pd.to_datetime(ohlc_df['timestamp'], utc=True)
            df = ohlc_df.copy()
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            if df.empty:
                self.logger.error("Input DataFrame is empty. Cannot compute features.")
                return None

            # Ensure required columns are present
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Column {col} is missing from input data.")
                    return None

            df = df.copy()
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Set 'timestamp' as index if necessary
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Compute features
        df = self.compute_technical_indicators(df)
        df = self.compute_vsa(df)
        df = self.detect_break_of_structure(df)
        df = self.identify_liquidity_zones(df)
        df = self.compute_multi_timeframe_features(df)
        df = self.detect_pullback_to_vwap(df)

        # Handle missing values
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        # Reset index to have 'timestamp' as a column
        df.reset_index(inplace=True)

        # Optionally, save features to the database
        if self.db is not None and df is not None:
            self.db.insert_features_data(df)
            self.logger.info("Features computed and updated in the database.")

        return df

    def compute_technical_indicators(self, df):
        """
        Compute technical indicators and add them to the dataframe.
        """
        df = df.copy()

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Moving averages
        df['ma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['ma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        # ATR
        df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_14'] = df['atr_14'].fillna(method='ffill')

        # Engulfing candles
        df['engulfing'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) / 100  # Normalize to -1, 0, 1

        # VWAP calculations
        df['vwap_30m'] = self.calculate_vwap(df, window='30min')
        df['vwap_daily'] = self.calculate_vwap(df, period='1D')

        return df

    def calculate_vwap(self, df, window=None, period=None):
        """
        Calculate VWAP for both rolling and regular intervals.
        """
        df = df.copy()

        if window is not None:
            cum_vp = (df['close'] * df['volume']).rolling(window=window, min_periods=1).sum()
            cum_vol = df['volume'].rolling(window=window, min_periods=1).sum()
            vwap = cum_vp / cum_vol
            vwap = vwap.bfill()
        elif period is not None:
            df_resampled = df.resample(period).agg({
                'close': 'last',
                'volume': 'sum',
                'high': 'max',
                'low': 'min',
                'open': 'first'
            })
            df_resampled['cum_volume_price'] = (df_resampled['close'] * df_resampled['volume']).cumsum()
            df_resampled['cum_volume'] = df_resampled['volume'].cumsum()
            vwap = df_resampled['cum_volume_price'] / df_resampled['cum_volume']
            vwap = vwap.reindex(df.index, method='ffill')
            vwap = vwap.bfill()
        else:
            raise ValueError("Either 'window' or 'period' must be specified.")

        return vwap

    def compute_vsa(self, df, sma_length=30, multipliers=(0.5, 1.5, 3.0)):
        """
        Compute Volume Spread Analysis (VSA) features.
        """
        df = df.copy()

        multiplier_1, multiplier_2, multiplier_3 = multipliers

        # Calculate Volume SMA with appropriate min_periods
        df['volume_sma'] = df['volume'].rolling(window=sma_length, min_periods=1).mean()

        # Handle initial NaN values by backfilling
        df['volume_sma'] = df['volume_sma'].bfill()

        # Calculate Cloud Thresholds
        df['cloud_1'] = df['volume_sma'] * multiplier_1
        df['cloud_2'] = df['volume_sma'] * multiplier_2
        df['cloud_3'] = df['volume_sma'] * multiplier_3

        # Categorize Volume
        conditions = [
            df['volume'] > df['cloud_3'],
            df['volume'] > df['cloud_2'],
            df['volume'] > df['cloud_1']
        ]
        choices = [3, 2, 1]  # High, Medium, Low volume categories
        df['vsa_category'] = np.select(conditions, choices, default=0)  # 0 for very low volume

        # Clean up intermediate columns if needed
        df.drop(['volume_sma', 'cloud_1', 'cloud_2', 'cloud_3'], axis=1, inplace=True)

        return df

    def detect_break_of_structure(self, df, significant_move_pct=1.0):
        """
        Identify Break of Structure (BOS) points in the market.
        """
        df = df.copy()

        # Calculate shifts
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)

        # Replace zero or NaN values to prevent division by zero
        df['prev_high'] = df['prev_high'].replace(0, np.nan)
        df['prev_low'] = df['prev_low'].replace(0, np.nan)

        # Calculate percentage change for significance
        df['high_change_pct'] = ((df['high'] - df['prev_high']) / df['prev_high']) * 100
        df['low_change_pct'] = ((df['low'] - df['prev_low']) / df['prev_low']) * 100

        # Handle NaN values
        df['high_change_pct'] = df['high_change_pct'].fillna(0)
        df['low_change_pct'] = df['low_change_pct'].fillna(0)

        # Identify higher highs/lows and lower highs/lows (used internally)
        higher_high = (df['high'] > df['prev_high']) & (df['high_change_pct'] > significant_move_pct)
        higher_low = (df['low'] > df['prev_low']) & (df['low_change_pct'] > significant_move_pct)
        lower_high = (df['high'] < df['prev_high']) & (df['high_change_pct'].abs() > significant_move_pct)
        lower_low = (df['low'] < df['prev_low']) & (df['low_change_pct'].abs() > significant_move_pct)

        # Determine uptrend and downtrend
        uptrend = higher_high & higher_low
        downtrend = lower_high & lower_low

        # Identify BOS with confirmation logic
        df['bos'] = ((uptrend & (~uptrend.shift(1).fillna(False))) |
                     (downtrend & (~downtrend.shift(1).fillna(False))))
        df['bos'] = df['bos'].astype(int)

        # Clean up intermediate columns
        df.drop(['prev_high', 'prev_low', 'high_change_pct', 'low_change_pct'], axis=1, inplace=True)

        return df

    def identify_liquidity_zones(self, df, window=20, threshold_pct=1.0, volume_factor=2.0):
        """
        Identify potential liquidity zones based on recent highs, lows, and volume.
        """
        df = df.copy()

        # Calculate rolling swing highs and lows with appropriate min_periods
        df['swing_high'] = df['high'].rolling(window=window, min_periods=1).max()
        df['swing_low'] = df['low'].rolling(window=window, min_periods=1).min()

        # Handle NaN values
        df['swing_high'] = df['swing_high'].bfill()
        df['swing_low'] = df['swing_low'].bfill()

        # Define thresholds for liquidity zones
        df['high_threshold'] = df['swing_high'] * (1 - threshold_pct / 100)
        df['low_threshold'] = df['swing_low'] * (1 + threshold_pct / 100)

        # Identify zones based on close proximity to thresholds
        df['near_high'] = (df['close'] >= df['high_threshold']) & (df['close'] <= df['swing_high'])
        df['near_low'] = (df['close'] <= df['low_threshold']) & (df['close'] >= df['swing_low'])

        # Volume confirmation for liquidity zones
        avg_volume = df['volume'].rolling(window=window, min_periods=1).mean()
        avg_volume = avg_volume.bfill()
        df['high_volume'] = df['volume'] >= avg_volume * volume_factor

        # Combine conditions to identify liquidity zones
        df['liquidity_zone'] = np.where(
            ((df['near_high'] | df['near_low']) & df['high_volume']),
            1, 0
        )

        # Clean up intermediate columns
        df.drop(['swing_high', 'swing_low', 'high_threshold', 'low_threshold',
                 'near_high', 'near_low', 'high_volume'], axis=1, inplace=True)

        return df

    def compute_multi_timeframe_features(self, df):
        """
        Compute VSA, BOS, and Liquidity Zones over multiple timeframes.
        """
        df = df.copy()

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Define the timeframes you want to compute features for
        timeframes = ['5min', '15min', '1H']  # 5-minute, 15-minute, 1-hour intervals

        for tf in timeframes:
            # Resample data
            df_tf = df.resample(tf).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # Compute VSA
            df_tf = self.compute_vsa(df_tf, sma_length=30)

            # Compute BOS
            df_tf = self.detect_break_of_structure(df_tf, significant_move_pct=1.0)

            # Identify Liquidity Zones
            df_tf = self.identify_liquidity_zones(df_tf, window=20, threshold_pct=1.0, volume_factor=2.0)

            # Forward-fill to align with the main dataframe
            df_tf = df_tf.reindex(df.index, method='ffill')

            # Append the features to the main dataframe with suffix indicating the timeframe
            tf_label = tf.lower()
            df[f'vsa_category_{tf_label}'] = df_tf['vsa_category']
            df[f'bos_{tf_label}'] = df_tf['bos']
            df[f'liquidity_zone_{tf_label}'] = df_tf['liquidity_zone']

        return df

    def detect_pullback_to_vwap(self, df):
        """
        Detect pullbacks to VWAP for fast-paced markets.
        """
        df = df.copy()

        buffer_pct = 0.001  # 0.1% buffer around VWAP
        avg_volume = df['volume'].rolling(window=5, min_periods=1).mean()
        avg_volume = avg_volume.bfill()
        df['vwap_30m_slope'] = df['vwap_30m'].diff()
        df['vwap_daily_slope'] = df['vwap_daily'].diff()
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()

        # Handle NaN values
        df['vwap_30m_slope'] = df['vwap_30m_slope'].fillna(0)
        df['vwap_daily_slope'] = df['vwap_daily_slope'].fillna(0)
        df['ema_5'] = df['ema_5'].bfill()

        # Pullback to 30-minute VWAP
        pullback_30m = (
            (df['low'] <= df['vwap_30m'] * (1 + buffer_pct)) &
            (df['low'] >= df['vwap_30m'] * (1 - buffer_pct)) &
            (df['close'] > df['vwap_30m']) &
            (df['volume'] > avg_volume) &
            (df['vwap_30m_slope'] > 0) &
            (df['close'] > df['ema_5'])
        )

        # Pullback to 1-day VWAP
        pullback_daily = (
            (df['low'] <= df['vwap_daily'] * (1 + buffer_pct)) &
            (df['low'] >= df['vwap_daily'] * (1 - buffer_pct)) &
            (df['close'] > df['vwap_daily']) &
            (df['volume'] > avg_volume) &
            (df['vwap_daily_slope'] > 0) &
            (df['close'] > df['ema_5'])
        )

        df['pullback_to_vwap_30m'] = pullback_30m.astype(int)
        df['pullback_to_vwap_daily'] = pullback_daily.astype(int)

        # Clean up intermediate columns
        df.drop(['vwap_30m_slope', 'vwap_daily_slope', 'ema_5'], axis=1, inplace=True)

        return df

    def get_historical_features(self, sequence_length):
        """
        Retrieve the last 'sequence_length' data points with features.

        :param sequence_length: Number of past data points to retrieve.
        :return: DataFrame with features.
        """
        if self.db is None:
            self.logger.error("Database interface not provided. Cannot fetch data.")
            return None

        df = self.db.get_features_data()
        df.sort_values('timestamp', inplace=True)
        # Take the last 'sequence_length' rows
        df = df.tail(sequence_length)
        # Handle missing values if necessary
        df.fillna(method='ffill', inplace=True)
        df.dropna(inplace=True)
        return df

    def compute_latest_features(self, latest_market_data):
        """
        Compute features for the latest market data point.

        :param latest_market_data: Series containing the latest market data.
        :return: Series with computed features.
        """
        # Convert latest_market_data to DataFrame
        df = latest_market_data.to_frame().T

        # Ensure 'timestamp' is present and set as index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
        else:
            self.logger.error("Latest market data must include 'timestamp'.")
            return None

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Compute features
        df_with_features = self.compute_features(df)

        # Return the latest features
        latest_features = df_with_features.iloc[-1]
        return latest_features

    def get_latest_features(self):
        """
        Retrieve the latest computed features for real-time prediction.
        """
        if self.db is None:
            self.logger.error("Database interface not provided. Cannot fetch data.")
            return None

        # Retrieve the latest data point
        latest_ohlc = self.db.get_latest_ohlc_data()
        latest_ohlc['timestamp'] = pd.to_datetime(latest_ohlc['timestamp'], utc=True)

        # Prepare DataFrame
        df = latest_ohlc.copy()
        df.sort_values('timestamp', inplace=True)
        df.set_index('timestamp', inplace=True)

        # Compute features
        df_with_features = self.compute_features(df)

        # Return the latest features
        latest_features = df_with_features.iloc[-1]
        return latest_features
