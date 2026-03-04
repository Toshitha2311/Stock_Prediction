import pandas as pd
import numpy as np
import os

def add_technical_indicators(df: pd.DataFrame):
    df = df.copy()

    # Moving Averages
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Financial Features
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Rolling_Volatility"] = df["Daily_Return"].rolling(20).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(10)

    df.dropna(inplace=True)
    return df


def create_lag_features(df: pd.DataFrame, lags=5):
    df = df.copy()

    for i in range(1, lags + 1):
        df[f"lag_{i}"] = df["Close"].shift(i)

    df.dropna(inplace=True)
    return df


def save_processed_data(df, symbol, period):
    os.makedirs("data/processed", exist_ok=True)
    file_path = f"data/processed/{symbol}_{period}_processed.csv"
    df.to_csv(file_path, index=False)