import pandas as pd
import numpy as np
from database.timescaledb_interface import TimescaleDBInterface
from sklearn.preprocessing import MinMaxScaler

def normalize_data(df, features):
    """
    Normalize data using MinMax scaling.
    """
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, scaler

def create_windowed_data(df, features, sequence_length):
    """
    Prepare windowed data sequences.
    """
    X = []
    y = []
    for i in range(len(df) - sequence_length):
        X.append(df[features].iloc[i:i+sequence_length].values)
        y.append(df['close'].iloc[i+sequence_length])
    X = np.array(X)
    y = np.array(y)
    return X, y

def mark_events(df):
    """
    Label data points with significant news events.
    """
    df['event_mark'] = np.where(df['sentiment_score'].abs() > 0.5, 1, 0)
    return df