import numpy as np
import pandas as pd

class TradingStrategy:
    def __init__(self, ai_model, feature_engineer):
        """
        Initialize the TradingStrategy with trained models and feature engineer.
        """
        self.ai_model = ai_model
        self.feature_engineer = feature_engineer
        self.historical_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # def generate_signal(self, latest_market_data):
    #     """
    #     Generate a trading signal based on the latest market data.
    #     """
    #     # Feature Engineering
    #     features = self.feature_engineer.compute_latest_features(latest_market_data)

    #     if features is None:
    #         return 'hold'

    #     # Get current price
    #     current_price = features['close']

    #     # XGBoost Prediction
    #     xgb_pred, xgb_proba = self.ai_model.predict(features)

    #     # LSTM Prediction
    #     lstm_input = self.prepare_lstm_input()
    #     if lstm_input is not None:
    #         lstm_pred = self.ai_model.lstm_model.predict(lstm_input)[0][0]  # Assuming output is a single value
    #     else:
    #         lstm_pred = current_price  # Default to current price if LSTM input is not available

    #     # PPO Agent Action
    #     ppo_obs = self.prepare_ppo_input(features)
    #     ppo_action, _ = self.ai_model.ppo_model.predict(ppo_obs)

    #     # Ensemble Decision
    #     signal = self.ensemble_decision(xgb_pred, xgb_proba, lstm_pred, ppo_action, current_price)

    #     return signal


    ###############################################################################################################
    #FOR BACKTESTING PURPOSES
     
    def generate_signal(self, latest_market_data):
        # Append the latest data to historical data
        self.historical_data = pd.concat(
            [self.historical_data, latest_market_data], ignore_index=True
        )

        # Compute features on historical data
        features_df = self.feature_engineer.compute_features_from_data(
            self.historical_data.copy()
        )

        if features_df is None or features_df.empty:
            self.feature_engineer.logger.info("No features computed, holding position.")
            return 'hold'

        # Use the last row as the current features
        current_features = features_df.iloc[-1]

        # Get current price
        current_price = current_features['close']

        # Prepare features for XGBoost
        feature_names = self.ai_model.scaler.feature_names_in_
        xgb_features = current_features[feature_names]

        # Ensure xgb_features is a DataFrame
        if isinstance(xgb_features, pd.Series):
            xgb_features = xgb_features.to_frame().T

        # XGBoost Prediction
        xgb_pred, xgb_proba = self.ai_model.predict(xgb_features)

        # LSTM Prediction
        lstm_input = self.prepare_lstm_input(features_df)
        if lstm_input is not None and self.ai_model.lstm_model is not None:
            lstm_pred = self.ai_model.lstm_model.predict(lstm_input)[0][0]
        else:
            lstm_pred = current_price

        # PPO Agent Action
        ppo_obs = self.prepare_ppo_input(current_features)
        if self.ai_model.ppo_model is not None:
            ppo_action, _ = self.ai_model.ppo_model.predict(ppo_obs)
        else:
            ppo_action = 0  # Default action if PPO model is not available

        # Ensemble Decision
        signal = self.ensemble_decision(xgb_pred, xgb_proba, lstm_pred, ppo_action, current_price)
        self.feature_engineer.logger.info(f"Ensemble signal: {signal}")
        return signal

#############################################################################################################################
    def prepare_lstm_input(self, features_df):
        """
        Prepare input for LSTM model.
        """
        # Retrieve the last sequence_length data points
        sequence_length = 30  # Must match the sequence length used during training
        df = features_df.tail(sequence_length)

        if df is None or len(df) < sequence_length:
            self.feature_engineer.logger.info("Insufficient historical data for LSTM input, using default value.")
            return None

        # Ensure features are in the correct order
        feature_columns = self.ai_model.lstm_feature_columns
        X = df[feature_columns].values.astype(np.float32)

        # Feature Scaling
        X_scaled = self.ai_model.scaler_lstm.transform(X)

        # Reshape for LSTM input
        X_sequence = X_scaled.reshape(1, sequence_length, len(feature_columns))

        return X_sequence


    def prepare_ppo_input(self, features):
        """
        Prepare input for PPO Agent.
        """
        # Exclude 'close' and 'timestamp' from observations
        ppo_obs = features.drop(['timestamp', 'close'], errors='ignore').values.astype(np.float32)
        return ppo_obs

    def ensemble_decision(self, xgb_pred, xgb_proba, lstm_pred, ppo_action, current_price):
        # Assign weights to each model
        weights = {
            'xgb': 0.4,
            'lstm': 0.3,
            'ppo': 0.3
        }

        # Initialize scores
        scores = {
            'buy': 0,
            'sell': 0,
            'hold': 0
        }

        # XGBoost contributes to scores
        xgb_signal = xgb_pred  # -1, 0, or 1
        if xgb_signal == 1:
            scores['buy'] += weights['xgb'] * xgb_proba[2]
        elif xgb_signal == -1:
            scores['sell'] += weights['xgb'] * xgb_proba[0]
        else:
            scores['hold'] += weights['xgb'] * xgb_proba[1]

        # LSTM prediction contributes to scores
        # Assuming lstm_pred is a predicted price
        if lstm_pred > current_price:
            scores['buy'] += weights['lstm']
        elif lstm_pred < current_price:
            scores['sell'] += weights['lstm']
        else:
            scores['hold'] += weights['lstm']

        # PPO Agent contributes to scores
        if ppo_action == 1:  # Buy
            scores['buy'] += weights['ppo']
        elif ppo_action == 2:  # Sell
            scores['sell'] += weights['ppo']
        else:  # Hold or Close Position
            scores['hold'] += weights['ppo']

        # Determine final signal
        final_signal = max(scores, key=scores.get)
        return final_signal
