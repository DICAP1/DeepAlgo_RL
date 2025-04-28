import talib
import pandas as pd
import numpy as np  # ✅ Ensure NumPy is imported


def compute_indicators(df):
    """Compute technical indicators including Bollinger Bands"""

    print("DEBUG: DataFrame Columns Before Indicator Calculation:", df.columns)
    print("DEBUG: DataFrame Sample Before Processing:\n", df.head())

    required_columns = ["Close", "High", "Low"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"❌ ERROR: Missing columns in DataFrame: {missing_columns}")
        print("DEBUG: DataFrame Sample (Missing Data):\n", df.head())
        return df

    close_prices = df["Close"].astype(np.float64).values
    high_prices = df["High"].astype(np.float64).values
    low_prices = df["Low"].astype(np.float64).values

    try:
        # ➕ Standard indicators
        df["EMA_50"] = talib.EMA(close_prices, timeperiod=50)
        df["EMA_200"] = talib.EMA(close_prices, timeperiod=200)
        df["RSI"] = talib.RSI(close_prices, timeperiod=14)
        df["ATR"] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)

        macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        df["MACD_Line"] = macd
        df["MACD_Signal"] = signal
        df["MACD_Histogram"] = hist

        # ➕ Bollinger Bands
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df["BB_Upper"] = upper
        df["BB_Middle"] = middle
        df["BB_Lower"] = lower

        # ➕ Additional indicators for reward logic
        df["RSI_6"] = talib.RSI(close_prices, timeperiod=6)
        df["ADX_20"] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=20)
        df["PLUS_DI_20"] = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=20)
        df["MINUS_DI_20"] = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=20)


        # 🔄 Fill NaNs with sensible defaults or rolling approximations
        df.fillna({
            "EMA_50": df["Close"].rolling(window=50, min_periods=1).mean(),
            "EMA_200": df["Close"].rolling(window=200, min_periods=1).mean(),
            "RSI": 50,
            "RSI_6": 50,
            "ADX_20": 20,
            "PLUS_DI_20": 20,
            "MINUS_DI_20": 20,
            "ATR": df["ATR"].median() if "ATR" in df else 0,
            "MACD_Line": 0,
            "MACD_Signal": 0,
            "MACD_Histogram": 0,
            "BB_Upper": df["Close"] + 0.01,
            "BB_Middle": df["Close"],
            "BB_Lower": df["Close"] - 0.01,
        }, inplace=True)


        df.fillna(0, inplace=True)

        print("✅ DEBUG: NaN values replaced in indicators.")
        print("DEBUG: DataFrame Sample After Processing:\n", df.head())

    except Exception as e:
        print(f"❌ ERROR: Indicator computation failed: {e}")

    return df
