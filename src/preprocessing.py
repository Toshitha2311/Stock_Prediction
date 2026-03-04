import pandas as pd
import os

def preprocess_data(file_path):
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # Ensure numeric columns
    for col in ["Open","High","Low","Close","Adj Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    
    # Save processed
    processed_path = "data/processed"
    os.makedirs(processed_path, exist_ok=True)
    save_file = f"{processed_path}/{os.path.basename(file_path)}"
    df.to_csv(save_file)
    print(f"Saved processed data to {save_file}")
    return df