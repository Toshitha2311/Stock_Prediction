import joblib
import os

def predict_future(df, symbol, period):

    model_path = f"models/{symbol}_{period}_model.pkl"

    if not os.path.exists(model_path):
        raise Exception("Model not trained yet!")

    model = joblib.load(model_path)

    latest_data = df.drop(columns=["Date"]).iloc[-1:]
    predictions = model.predict(latest_data)

    return predictions[0]