import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model(df, target_days):
    df = df.copy()
    df["Target"] = df["Close"].shift(-target_days)
    df.dropna(inplace=True)

    X = df[["Close"]]
    y = df["Target"]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)

    return model, mse


def predict_future(model, df, target_days):
    last_close = df["Close"].iloc[-1]
    future_prices = []
    current_price = last_close

    for _ in range(target_days):
        next_price = model.predict([[current_price]])[0]
        future_prices.append(next_price)
        current_price = next_price

    return future_prices