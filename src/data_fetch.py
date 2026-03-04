import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(symbol="AAPL", period="1y", save_path="data/raw"):
    os.makedirs(save_path, exist_ok=True)
    df = yf.download(symbol, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    file_name = f"{save_path}/{symbol}_{period}.csv"
    df.to_csv(file_name)
    print(f"Saved raw data to {file_name}")
    return df