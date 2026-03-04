from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import numpy as np

def train_model(df, symbol, period, target_days=3):

    df = df.copy()

    # Create multi-output targets
    for i in range(1, target_days + 1):
        df[f"target_{i}"] = df["Close"].shift(-i)

    df.dropna(inplace=True)

    X = df.drop(columns=["Date"] + [f"target_{i}" for i in range(1, target_days + 1)])
    y = df[[f"target_{i}" for i in range(1, target_days + 1)]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{symbol}_{period}_model.pkl"
    joblib.dump(model, model_path)

    return model, mse, mae, r2