StockMovementPredictor/
│
├── data/
│   ├── raw/                 # Raw stock data from API
│   ├── processed/           # Cleaned & feature-engineered data
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── feature_engineering.ipynb
│
├── src/
│   ├── data_fetch.py        # Fetch data from API
│   ├── preprocessing.py     # Clean data, handle missing values
│   ├── features.py          # Technical indicators & lag features
│   ├── model_train.py       # Train multi-output ML model
│   ├── model_predict.py     # Predict N days ahead (user-defined)
│   ├── utils.py             # Helper functions
│
├── dashboard/
│   ├── powerbi/             # Power BI files + CSV for visualization
│
├── reports/
│   ├── evaluation/
│   ├── feature_importance/
│
├── requirements.txt
└── README.md