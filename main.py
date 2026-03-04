from src.data_fetch import fetch_stock_data
from src.preprocessing import preprocess_data
from src.model import train_model, predict_future

def main():
    print("📈 STOCK PREDICTION SYSTEM")
    print("-" * 40)

    # User Inputs
    symbol = input("Enter Stock Symbol (e.g., AAPL, TSLA, RELIANCE.NS): ").upper()
    period = input("Enter Time Period (1mo, 3mo, 6mo, 1y, 5y): ")
    target_days = int(input("How many days ahead do you want to predict? "))

    print("\nFetching data...")
    df = fetch_stock_data(symbol, period)

    print("Preprocessing data...")
    df = preprocess_data(df)

    print("Training model...")
    model, mse = train_model(df, symbol, period, target_days)

    print(f"\nModel trained successfully ✅")
    print(f"Mean Squared Error: {mse:.4f}")

    print("\nPredicting future prices...")
    future_prices = predict_future(model, df, target_days)

    print("\n📊 Predicted Prices:")
    for i, price in enumerate(future_prices, 1):
        print(f"Day {i}: {price:.2f}")

if __name__ == "__main__":
    main()