import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
import gym
import optuna
from gym import spaces
import joblib
from utils.logger import get_logger
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

class AIModel:
    """
    Class for AI model development and training.
    """

    def __init__(self, db_interface):
        self.db = db_interface
        self.logger = get_logger(self.__class__.__name__)

    def prepare_data_with_scaling(self):
        """
        Prepare data for model training with feature scaling and optional dimensionality reduction.
        """
        df = self.db.get_features_data()
        df.sort_values('timestamp', inplace=True)
        self.logger.info(f"Data loaded with shape: {df.shape}")

        # Define features
        features = [col for col in df.columns if col not in ['timestamp', 'signal', 'future_return']]
        self.features = features  # Save features for later use

        # Ensure all features are numeric
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Log NaN counts after conversion
        nan_counts = df[features].isna().sum()
        self.logger.info(f"NaN counts after conversion to numeric:\n{nan_counts}")

        # Drop columns with high NaN counts (e.g., more than 50% NaNs)
        threshold = 0.5  # 50% threshold
        max_na = threshold * len(df)
        cols_to_drop = nan_counts[nan_counts > max_na].index.tolist()
        self.logger.info(f"Dropping columns due to high NaN counts: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
        features = [col for col in features if col not in cols_to_drop]

        # Optionally, drop specific known problematic columns
        known_problematic_cols = ['higher_high', 'lower_low', 'bos', 'bos_5min', 'bos_15min', 'engulfing']
        df.drop(columns=known_problematic_cols, inplace=True, errors='ignore')
        features = [col for col in features if col not in known_problematic_cols]

        # Generate target signals if not present
        if 'signal' not in df.columns:
            df['signal'] = self.generate_signal(df)

        # Handle missing values by filling them with zero
        df.fillna(0, inplace=True)
        self.logger.info(f"Data shape after filling missing values: {df.shape}")

        # Ensure we have data after preprocessing
        if df.empty:
            self.logger.error("DataFrame is empty after preprocessing. Cannot proceed with training.")
            raise ValueError("DataFrame is empty after preprocessing.")

        # Features and target
        X = df[features]
        y = df['signal']

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save the scaler for future use
        self.scaler = scaler

        # Dimensionality Reduction (Optional)
        # You can adjust the n_components parameter as needed
        pca = PCA(n_components=0.95)  # Retain 95% of variance
        X_reduced = pca.fit_transform(X_scaled)

        # Save the PCA model for future use
        self.pca = pca

        # Train-Test Split using TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        train_indices, test_indices = list(tscv.split(X_reduced))[-1]
        X_train, X_test = X_reduced[train_indices], X_reduced[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        return X_train, X_test, y_train, y_test

    def prepare_lstm_data(self):
        """
        Prepare data for LSTM model training.
        """
        df = self.db.get_features_data()
        df.sort_values('timestamp', inplace=True)
        self.logger.info(f"Data loaded with shape: {df.shape}")

        # Define features
        features = [col for col in df.columns if col not in ['timestamp', 'signal', 'future_return']]
        self.lstm_feature_columns = features  # Save features for later use

        # Ensure all features are numeric
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Log NaN counts after conversion
        nan_counts = df[features].isna().sum()
        self.logger.info(f"NaN counts after conversion to numeric:\n{nan_counts}")

        # Drop columns with high NaN counts
        threshold = 0.5
        max_na = threshold * len(df)
        cols_to_drop = nan_counts[nan_counts > max_na].index.tolist()
        self.logger.info(f"Dropping columns due to high NaN counts: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
        features = [col for col in features if col not in cols_to_drop]

        # Drop known problematic columns
        known_problematic_cols = ['higher_high', 'lower_low', 'bos', 'bos_5min', 'bos_15min', 'engulfing']
        df.drop(columns=known_problematic_cols, inplace=True, errors='ignore')
        features = [col for col in features if col not in known_problematic_cols]

        # Generate target signals if not present
        if 'signal' not in df.columns:
            df['signal'] = self.generate_signal(df)

        # Handle missing values
        df.fillna(0, inplace=True)
        self.logger.info(f"Data shape after filling missing values: {df.shape}")

        # Ensure we have data after preprocessing
        if df.empty:
            self.logger.error("DataFrame is empty after preprocessing. Cannot proceed with training.")
            raise ValueError("DataFrame is empty after preprocessing.")

        # Features and target
        X = df[features].values.astype(np.float32)
        y = df['signal'].values

        # Feature Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Save the scaler for future use
        self.scaler_lstm = scaler

        # Reshape for LSTM input
        sequence_length = 30  # 30 time steps
        X_sequences = []
        y_sequences = []
        for i in range(len(X_scaled) - sequence_length):
            X_seq = X_scaled[i:i + sequence_length]
            y_seq = y[i + sequence_length]
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        # Train-Test Split using TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        train_indices, test_indices = list(tscv.split(X_sequences))[-1]
        X_train, X_test = X_sequences[train_indices], X_sequences[test_indices]
        y_train, y_test = y_sequences[train_indices], y_sequences[test_indices]

        return X_train, X_test, y_train, y_test

    def build_lstm_model(self, input_shape):
        """
        Build and compile the LSTM model.
        """
        model = Sequential()
        model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_lstm(self):
        """
        Train the LSTM model.
        """
        X_train, X_test, y_train, y_test = self.prepare_lstm_data()
        model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test))
        self.lstm_model = model
        self.logger.info("LSTM model trained successfully.")

    def train_xgboost_with_scaling(self):
        """
        Train the XGBoost model with scaled features and dimensionality reduction.
        """
        X_train, X_test, y_train, y_test = self.prepare_data_with_scaling()

        # Map signals to labels 0,1,2 for XGBoost
        label_mapping = {-1: 0, 0: 1, 1: 2}
        y_train_mapped = y_train.map(label_mapping)
        y_test_mapped = y_test.map(label_mapping)

        # Initialize and train model
        xgb_model = XGBClassifier(
            use_label_encoder=False,
            objective='multi:softmax',
            num_class=3,
            eval_metric='mlogloss',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        xgb_model.fit(X_train, y_train_mapped)

        # Feature Importance
        importance = xgb_model.feature_importances_
        feature_names = [f'PC{i+1}' for i in range(X_train.shape[1])]
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df.sort_values(by='Importance', ascending=False, inplace=True)
        self.logger.info(f"Feature Importances:\n{importance_df}")

        # Optionally select top N features based on importance
        # For example, keep top 20 features
        top_features = importance_df['Feature'].iloc[:20].tolist()
        top_indices = [int(f[2:]) - 1 for f in top_features]  # Extract indices from 'PC1', 'PC2', etc.
        self.top_indices = top_indices  # Save indices for future use
        X_train_top = X_train[:, top_indices]
        X_test_top = X_test[:, top_indices]

        # Retrain model with top features
        xgb_model.fit(X_train_top, y_train_mapped)

        # Evaluate model
        y_pred_mapped = xgb_model.predict(X_test_top)
        # Map labels back to original signals
        inv_label_mapping = {v: k for k, v in label_mapping.items()}
        y_pred = pd.Series(y_pred_mapped).map(inv_label_mapping)
        y_test_original = y_test.reset_index(drop=True)
        report = classification_report(y_test_original, y_pred)
        self.logger.info(f"XGBoost Model Performance with Top Features:\n{report}")

        self.xgb_model = xgb_model

    def train_ppo_agent(self):
        """
        Train the PPO reinforcement learning agent.
        """
        # Create a custom environment
        env = DummyVecEnv([lambda: TradingEnv(self.db)])

        # Wrap the environment with VecNormalize for observation normalization
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

        # Define the PPO model
        model = PPO('MlpPolicy', env, verbose=1)
        
        # Train the model
        # model.learn(total_timesteps=100000)
        model.learn(total_timesteps=10000)
        
        # Save the model
        self.ppo_model = model
        self.logger.info("PPO agent trained successfully.")

    def generate_signal(self, df, threshold=0.001):
        """
        Generate trading signals based on future returns.

        Parameters:
        - df: DataFrame with 'close' price.
        - threshold: Minimum return to consider a significant movement.

        Returns:
        - signals: Series with signals (-1, 0, 1).
        """
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        df['signal'] = 0  # Default to hold
        df.loc[df['future_return'] > threshold, 'signal'] = 1  # Buy
        df.loc[df['future_return'] < -threshold, 'signal'] = -1  # Sell
        return df['signal']

    def save_model(self, model_name=None):
        """
        Save the trained models and preprocessing objects.
        """
        os.makedirs('models', exist_ok=True)

        # Save LSTM model
        if model_name == 'lstm' or model_name is None:    
            if hasattr(self, 'lstm_model'):
                self.lstm_model.save('models/lstm_model.h5')
                self.logger.info("LSTM model saved successfully.")

        # Save XGBoost model
        if model_name == 'xgboost' or model_name is None:    
            if hasattr(self, 'xgb_model'):
                joblib.dump(self.xgb_model, 'models/xgboost_model.joblib')
                self.logger.info("XGBoost model saved successfully.")

        # Save PPO model
        if model_name == 'ppo' or model_name is None:    
            if hasattr(self, 'ppo_model'):
                self.ppo_model.save('models/ppo_agent')
                self.logger.info("PPO agent saved successfully.")

        # Save Scaler
        if hasattr(self, 'scaler'):
            joblib.dump(self.scaler, 'models/scaler.joblib')
            self.logger.info("Scaler saved successfully.")

        # Save PCA
        if hasattr(self, 'pca'):
            joblib.dump(self.pca, 'models/pca.joblib')
            self.logger.info("PCA model saved successfully.")

        # Save top feature indices
        if hasattr(self, 'top_indices'):
            joblib.dump(self.top_indices, 'models/top_indices.joblib')
            self.logger.info("Top feature indices saved successfully.")

        # Save LSTM scaler
        if hasattr(self, 'scaler_lstm'):
            joblib.dump(self.scaler_lstm, 'models/scaler_lstm.joblib')
            self.logger.info("LSTM scaler saved successfully.")

        # Save LSTM feature columns
        if hasattr(self, 'lstm_feature_columns'):
            joblib.dump(self.lstm_feature_columns, 'models/lstm_feature_columns.joblib')
            self.logger.info("LSTM feature columns saved successfully.")

    def load_model(self):
        """
        Load the trained models and preprocessing objects.
        """
        # Load LSTM model
        try:
            self.lstm_model = load_model('models/lstm_model.h5')
            self.logger.info("LSTM model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading LSTM model: {e}")

        # Load XGBoost model
        try:
            self.xgb_model = joblib.load('models/xgboost_model.joblib')
            self.logger.info("XGBoost model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading XGBoost model: {e}")

        # Load PPO model
        try:
            self.ppo_model = PPO.load('models/ppo_agent')
            self.logger.info("PPO agent loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading PPO agent: {e}")

        # Load Scaler
        try:
            self.scaler = joblib.load('models/scaler.joblib')
            self.logger.info("Scaler loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading scaler: {e}")

        # Load PCA
        try:
            self.pca = joblib.load('models/pca.joblib')
            self.logger.info("PCA model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading PCA model: {e}")

        # Load top feature indices
        try:
            self.top_indices = joblib.load('models/top_indices.joblib')
            self.logger.info("Top feature indices loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading top feature indices: {e}")

        # Load LSTM scaler
        try:
            self.scaler_lstm = joblib.load('models/scaler_lstm.joblib')
            self.logger.info("LSTM scaler loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading LSTM scaler: {e}")

        # Load LSTM feature columns
        try:
            self.lstm_feature_columns = joblib.load('models/lstm_feature_columns.joblib')
            self.logger.info("LSTM feature columns loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading LSTM feature columns: {e}")

    def predict(self, latest_features):
        """
        Make predictions using the trained models.
        """
        # Prepare data
        X = latest_features.drop(['timestamp', 'signal', 'future_return'], errors='ignore').values.reshape(1, -1)

        # Feature Scaling
        X_scaled = self.scaler.transform(X)

        # Dimensionality Reduction
        X_reduced = self.pca.transform(X_scaled)

        # Select top features
        X_top = X_reduced[:, self.top_indices]

        # Make prediction with XGBoost
        y_pred_mapped = self.xgb_model.predict(X_top)
        y_pred_proba = self.xgb_model.predict_proba(X_top)

        inv_label_mapping = {0: -1, 1: 0, 2: 1}
        y_pred = inv_label_mapping.get(y_pred_mapped[0], 0)
        return y_pred, y_pred_proba[0]

class TradingEnv(gym.Env):
    """
    Custom Environment for PPO Agent.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, db_interface):
        super(TradingEnv, self).__init__()
        self.db = db_interface
        # Fetch features data
        self.data = self.db.get_features_data()
        self.logger = get_logger(self.__class__.__name__)
        self._preprocess_data()
        self.n_features = len(self.data.columns) - 2  # Exclude 'close' or 'timestamp' if necessary
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 0: Hold, 1: Buy, 2: Sell, 3: Close Position
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)
        self.current_step = 0
        self.position = 0  # 0: Flat, 1: Long, -1: Short
        self.entry_price = 0.0
        self.total_profit = 0.0
        self.history = {
            'timestamp': [],
            'price': [],
            'action': [],
            'position': [],
            'profit': [],
            'total_profit': []
        }

    def _preprocess_data(self):
        df = self.data
        df.sort_values('timestamp', inplace=True)

        # Define features
        features = [col for col in df.columns if col not in ['timestamp', 'close', 'signal', 'future_return']]
        self.features = features  # Save features

        # Ensure all features are numeric
        for col in features:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Log NaN counts after conversion
        nan_counts = df[features].isna().sum()
        self.logger.info(f"NaN counts in environment data after conversion to numeric:\n{nan_counts}")

        # Drop columns with high NaN counts
        threshold = 0.5
        max_na = threshold * len(df)
        cols_to_drop = nan_counts[nan_counts > max_na].index.tolist()
        self.logger.info(f"Dropping columns in environment due to high NaN counts: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
        features = [col for col in features if col not in cols_to_drop]

        # Drop known problematic columns
        known_problematic_cols = ['higher_high', 'lower_low', 'bos', 'bos_5min', 'bos_15min', 'engulfing']
        df.drop(columns=known_problematic_cols, inplace=True, errors='ignore')
        features = [col for col in features if col not in known_problematic_cols]

        # Handle missing values
        df.fillna(0, inplace=True)
        self.logger.info(f"Environment data shape after preprocessing: {df.shape}")

        # Update self.data and self.features
        self.data = df
        self.features = features

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.total_profit = 0.0
        # Reset history
        self.history = {
            'timestamp': [],
            'price': [],
            'action': [],
            'position': [],
            'profit': [],
            'total_profit': []
        }
        return self._next_observation()

    def step(self, action):
        done = False

        # Get current price
        current_price = self.data.iloc[self.current_step]['close']

        # Update portfolio
        reward = 0
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
        elif action == 3:  # Close Position
            if self.position != 0:
                # Calculate profit or loss
                price_change = current_price - self.entry_price
                if self.position == 1:
                    reward = price_change
                elif self.position == -1:
                    reward = -price_change
                self.total_profit += reward
                self.position = 0
                self.entry_price = 0.0
        else:  # Hold
            # Calculate unrealized profit or loss
            if self.position != 0:
                price_change = current_price - self.entry_price
                if self.position == 1:
                    reward = price_change
                elif self.position == -1:
                    reward = -price_change

        # Record history
        self.history['timestamp'].append(self.data.iloc[self.current_step]['timestamp'])
        self.history['price'].append(current_price)
        self.history['action'].append(action)
        self.history['position'].append(self.position)
        self.history['profit'].append(reward)
        self.history['total_profit'].append(self.total_profit)

        # Proceed to next step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        # Ensure reward is finite
        if not np.isfinite(reward):
            self.logger.error(f"Reward is not finite at step {self.current_step}. Reward: {reward}")
            reward = 0.0  # Set to zero or handle appropriately

        obs = self._next_observation()

        return obs, reward, done, {}

    def _next_observation(self):
        # Get next observation excluding 'close' and 'timestamp' columns
        obs = self.data.iloc[self.current_step][self.features].values.astype(np.float32)
        if not np.isfinite(obs).all():
            self.logger.error(f"Observation contains NaNs or Infs at step {self.current_step}")
            self.logger.error(f"Observation: {obs}")
            raise ValueError("Observation contains NaNs or Infs")
        return obs

    def calculate_performance_metrics(self):
        """
        Calculate performance metrics like total return, maximum drawdown, and Sharpe ratio.
        """
        profits = np.array(self.history['profit'])
        cumulative_profits = np.array(self.history['total_profit'])

        # Total Return
        total_return = cumulative_profits[-1] if len(cumulative_profits) > 0 else 0.0

        # Sharpe Ratio
        if np.std(profits) != 0 and len(profits) > 1:
            sharpe_ratio = (np.mean(profits) / np.std(profits)) * np.sqrt(252)  # Adjust if needed
        else:
            sharpe_ratio = 0.0

        # Maximum Drawdown
        if len(cumulative_profits) > 0:
            running_max = np.maximum.accumulate(cumulative_profits)
            drawdowns = running_max - cumulative_profits
            max_drawdown = np.max(drawdowns)
        else:
            max_drawdown = 0.0

        performance = {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown
        }

        return performance

    def render(self, mode='human', save_fig=False, filename='trading_performance.html'):
        # Calculate performance metrics
        performance = self.calculate_performance_metrics()

        # Convert timestamps to datetime if they are not already
        timestamps = pd.to_datetime(self.history['timestamp'], utc=True)

        # Create subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.02,
                            subplot_titles=('Price and Agent Actions', 'Agent Position Over Time', 'Cumulative Profit Over Time'))

        # Price and Actions
        fig.add_trace(go.Scatter(x=timestamps, y=self.history['price'], mode='lines', name='Price', line=dict(color='blue')), row=1, col=1)

        # Buy, Sell, Close markers
        actions = np.array(self.history['action'])
        buy_indices = np.where(actions == 1)[0]
        sell_indices = np.where(actions == 2)[0]
        close_indices = np.where(actions == 3)[0]

        fig.add_trace(go.Scatter(x=timestamps[buy_indices], y=np.array(self.history['price'])[buy_indices],
                                mode='markers', marker_symbol='triangle-up', marker_color='green', marker_size=10,
                                name='Buy'), row=1, col=1)

        fig.add_trace(go.Scatter(x=timestamps[sell_indices], y=np.array(self.history['price'])[sell_indices],
                                mode='markers', marker_symbol='triangle-down', marker_color='red', marker_size=10,
                                name='Sell'), row=1, col=1)

        fig.add_trace(go.Scatter(x=timestamps[close_indices], y=np.array(self.history['price'])[close_indices],
                                mode='markers', marker_symbol='circle', marker_color='black', marker_size=10,
                                name='Close'), row=1, col=1)

        # Agent Position
        fig.add_trace(go.Scatter(x=timestamps, y=self.history['position'], mode='lines', name='Position', line=dict(color='purple')), row=2, col=1)

        # Cumulative Profit
        fig.add_trace(go.Scatter(x=timestamps, y=self.history['total_profit'], mode='lines', name='Cumulative Profit', line=dict(color='orange')), row=3, col=1)

        # Update layout
        fig.update_layout(height=800, width=1000, title_text='Trading Performance')

        # Add performance metrics as annotations
        metrics_text = f"Total Return: {performance['Total Return']:.2f}<br>" \
                    f"Sharpe Ratio: {performance['Sharpe Ratio']:.2f}<br>" \
                    f"Maximum Drawdown: {performance['Maximum Drawdown']:.2f}"

        fig.add_annotation(
            text=metrics_text,
            xref="paper", yref="paper",
            x=0, y=-0.2, showarrow=False,
            align='left', bgcolor='orange', opacity=0.7
        )

        if save_fig:
            fig.write_html(filename)
            self.logger.info(f"Interactive plot saved as {filename}")

        fig.show()
