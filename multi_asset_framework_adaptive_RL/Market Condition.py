


import oandapyV20
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from arch import arch_model

# ---- OANDA API Credentials ----
OANDA_API_KEY = "f7eff581944bb0b5efb4cac08003be9d-feea72696d43fb03101ddaa84eea2148"
OANDA_ACCOUNT_ID = "101-004-1683826-005"
#OANDA_INSTRUMENTS = ["NZD_CAD","NZD_USD","EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD","US30_USD","USB05Y_USD","EU50_EUR"]

OANDA_INSTRUMENTS = [
    # Currencies
    'AUD_USD', 'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CAD', 'EUR_GBP',
    'EUR_JPY', 'GBP_JPY', 'NZD_USD', 'USD_CHF', 'USD_HKD', 'USD_SGD',
    'USD_MXN', 'USD_CNH', 'USD_NOK', 'USD_SEK', 'USD_TRY', 'USD_ZAR',
    'EUR_AUD', 'EUR_CAD', 'EUR_DKK', 'EUR_HUF', 'EUR_NOK', 'EUR_NZD',
    'EUR_PLN', 'EUR_SEK', 'EUR_TRY', 'GBP_AUD', 'GBP_CAD', 'GBP_CHF',
    'GBP_NZD', 'AUD_CAD', 'AUD_CHF', 'AUD_JPY', 'AUD_NZD', 'NZD_CAD',

    # Commodities
    'XAU_USD', 'XAG_USD', 'WTICO_USD', 'BCO_USD', 'NATGAS_USD',
    'XCU_USD', 'XPT_USD',

    # Indices
    'US30_USD', 'SPX500_USD', 'NAS100_USD', 'DE30_EUR', 'UK100_GBP',
    'JP225_USD', 'FR40_EUR', 'AU200_AUD', 'HK33_HKD', 'CH20_CHF',
    'EU50_EUR', 'NL25_EUR', 'SG30_SGD', 'ESPIX_EUR', 'CN50_USD',

    # Bonds
    'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD', 'DE10YB_EUR',
    'UK10YB_GBP',

    # Additional CFDs
    'CHINAH_HKD', 'US2000_USD', 'WHEAT_USD', 'SUGAR_USD', 'JP225Y_JPY',
    'CORN_USD', 'SOYBN_USD'
]


# Initialize OANDA API client
client = oandapyV20.API(access_token=OANDA_API_KEY)

# ---- Setup Logging ----
logging.basicConfig(
    filename="../trading_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logging.getLogger().addHandler(console_handler)

# ---- Fetch Real-Time Prices ----
def fetch_real_time_prices():
    """Fetch real-time bid/ask prices from OANDA for multiple instruments."""
    try:
        params = {"instruments": ",".join(OANDA_INSTRUMENTS)}
        r = pricing.PricingInfo(accountID=OANDA_ACCOUNT_ID, params=params)
        client.request(r)
        prices = r.response["prices"]
        return {price["instrument"]: (float(price["bids"][0]["price"]), float(price["asks"][0]["price"])) for price in prices}
    except Exception as e:
        logging.error(f"Error fetching real-time prices: {str(e)}")
        return {}






# Fetch real-time pricesreal_time_prices = fetch_real_time_prices()# Display the fetched dataif real_time_prices:    for instrument, (bid, ask) in real_time_prices.items():        print(f"{instrument} - Bid: {bid:.5f}, Ask: {ask:.5f}")else:    print("⚠️ No real-time prices fetched. Check API connectivity.")

# ---- Fetch Historical Data ----
def fetch_historical_data(instrument):
    """Fetches historical data from OANDA and ensures it is valid."""
    try:
        params = {"count": 500, "granularity": "M5"}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)
        candles = r.response.get("candles", [])

        if not candles or len(candles) < 50:
            logging.error(f"Insufficient historical data for {instrument}. Skipping.")
            return pd.DataFrame()

        data = []
        for candle in candles:
            if "mid" not in candle:
                continue
            data.append([
                float(candle["mid"]["o"]),
                float(candle["mid"]["h"]),
                float(candle["mid"]["l"]),
                float(candle["mid"]["c"]),
            ])

        df = pd.DataFrame(data, columns=["Open", "High", "Low", "Close"])
        return df

    except Exception as e:
        logging.error(f"Failed to fetch historical data for {instrument}: {str(e)}")
        return pd.DataFrame()


# ---- Compute Indicators ----
# ---- Compute Indicators (Manual Calculations) ----
def compute_indicators(df):
    """Computes technical indicators manually and ensures all required values are present."""
    if df.empty:
        logging.error("⚠️ Indicator computation skipped due to empty DataFrame.")
        return df

    try:
        # ✅ ATR (Average True Range)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        true_range = pd.DataFrame({"high_low": high_low, "high_close": high_close, "low_close": low_close})
        df["ATR"] = true_range.max(axis=1).rolling(window=14).mean()

        # ✅ RSI (Relative Strength Index)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # ✅ MACD (Moving Average Convergence Divergence)
        df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # ✅ Bollinger Bands
        rolling_mean = df["Close"].rolling(window=20).mean()
        rolling_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = rolling_mean + (rolling_std * 2)
        df["BB_Middle"] = rolling_mean
        df["BB_Lower"] = rolling_mean - (rolling_std * 2)

        # ✅ EMA (Exponential Moving Averages)
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
        df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()

        # ✅ ADX (Average Directional Index)
        plus_dm = pd.Series(np.where((df["High"] - df["High"].shift(1)) > (df["Low"].shift(1) - df["Low"]), df["High"] - df["High"].shift(1), 0), index=df.index)
        minus_dm = pd.Series(np.where((df["Low"].shift(1) - df["Low"]) > (df["High"] - df["High"].shift(1)), df["Low"].shift(1) - df["Low"], 0), index=df.index)

        tr14 = df["ATR"].rolling(window=14).sum()
        plus_di14 = (100 * (plus_dm.rolling(window=14).sum() / tr14)).fillna(0)
        minus_di14 = (100 * (minus_dm.rolling(window=14).sum() / tr14)).fillna(0)
        dx = (100 * np.abs((plus_di14 - minus_di14) / (plus_di14 + minus_di14))).fillna(0)
        df["ADX"] = dx.rolling(window=14).mean()

        # ✅ ROC (Rate of Change)
        df["ROC"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)) * 100

        # ✅ Fill Missing Values (Updated for FutureWarning)
        df.bfill(inplace=True)  # Backfill
        df.ffill(inplace=True)  # Forward Fill
        df.fillna(0, inplace=True)  # Replace any remaining NaNs with 0

        return df

    except Exception as e:
        logging.error(f"❌ Error computing indicators: {str(e)}")
        return df

    except Exception as e:
        logging.error(f"❌ Error computing indicators: {str(e)}")
        return df




# Loop through all instruments
for instrument in OANDA_INSTRUMENTS:
    print(f"\n🔹 Processing {instrument}...")  # Print which instrument is being processed

    # Fetch historical data
    df = fetch_historical_data(instrument)

    # Check if data was retrieved successfully
    if df.empty:
        print(f"⚠️ No historical data available for {instrument}. Skipping...\n")
        continue  # Skip to the next instrument if data is empty

    # Compute technical indicators
    df = compute_indicators(df)

    # Display the first 5 rows of the DataFrame
    print(df.head())  # Show only first 5 rows to confirm indicators

    print("-" * 50)  # Separator for readability



# ---- Trend Following Strategy (EMA Crossover + ADX Confirmation) ----
def trend_following_strategy(df):
    """Uses EMA 20 & EMA 50 Crossovers for trend following with ADX confirmation."""

    if df.empty or len(df) < 50:
        return "No Data Available"

    # ✅ Ensure required columns exist before accessing
    required_columns = ["EMA_20", "EMA_50", "ADX"]
    if not all(col in df.columns for col in required_columns):
        return "Indicator Data Missing"

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    # ✅ Trend confirmation using ADX (Avoid weak trends)
    if last_row["ADX"] < 20:
        return "No Strong Trend (Weak ADX)"

    if last_row["EMA_20"] > last_row["EMA_50"] and prev_row["EMA_20"] <= prev_row["EMA_50"]:
        return "Trend Following: Uptrend (Long Position)"

    elif last_row["EMA_20"] < last_row["EMA_50"] and prev_row["EMA_20"] >= prev_row["EMA_50"]:
        return "Trend Following: Downtrend (Short Position)"

    return "No Clear Trend"

# ---- Momentum Strategy ----
def momentum_strategy(df):
    """Uses Rate of Change (ROC) and RSI to confirm momentum trades."""
    if df.empty or len(df) < 10:
        return "No Data Available"

    if "ROC" not in df.columns:
        return "ROC Data Missing"

    last_row = df.iloc[-1]

    if last_row["ROC"] > 0 and last_row["RSI"] < 70:
        return "Momentum: Strong Upward (Long Position)"
    elif last_row["ROC"] < 0 and last_row["RSI"] > 30:
        return "Momentum: Strong Downward (Short Position)"

    return "No Momentum Signal"

# ---- MACD + RSI Strategy (with ADX Confirmation) ----
def macd_rsi_strategy(df):
    """Combines MACD and RSI for trade confirmation, with ADX validation."""
    if df.empty or len(df) < 26:  # Ensure enough data for MACD
        return "No Data Available"

    # ✅ Check if required indicators exist before using them
    required_columns = ["MACD", "MACD_Signal", "RSI", "ADX"]
    if not all(col in df.columns for col in required_columns):
        return "Indicator Data Missing"

    last_row = df.iloc[-1]

    # ✅ Ensure values are not NaN
    if pd.isna(last_row["MACD"]) or pd.isna(last_row["MACD_Signal"]) or pd.isna(last_row["RSI"]) or pd.isna(last_row["ADX"]):
        return "MACD/RSI Data Unavailable"

    # ✅ Trend strength validation using ADX (Avoid weak trends)
    if last_row["ADX"] < 20:
        return "No Strong Trend (Weak ADX)"

    # ✅ Trading Signals
    if last_row["MACD"] > last_row["MACD_Signal"] and last_row["RSI"] < 80:
        return "MACD + RSI: Bullish Signal (Long Position)"

    elif last_row["MACD"] < last_row["MACD_Signal"] and last_row["RSI"] > 20:
        return "MACD + RSI: Bearish Signal (Short Position)"

    return "No Clear Signal"


# ---- Pinbar Reversal Strategy ----
def pinbar_strategy(df):
    """Detects Pinbar candlestick patterns indicating reversals."""
    if df.empty or len(df) < 5:
        return "No Data Available"
    last_row = df.iloc[-1]
    body = abs(last_row["Open"] - last_row["Close"])
    upper_shadow = last_row["High"] - max(last_row["Open"], last_row["Close"])
    lower_shadow = min(last_row["Open"], last_row["Close"]) - last_row["Low"]
    if upper_shadow > 2 * body and lower_shadow < body:
        return "Bearish Pinbar Reversal (Sell)"
    elif lower_shadow > 2 * body and upper_shadow < body:
        return "Bullish Pinbar Reversal (Buy)"
    return "No Pinbar Pattern"
# ---- Volatility-Based Adaptive Thresholds (ATR) ----
def adaptive_thresholds(df):
    """Adjusts RSI and ADX thresholds dynamically based on ATR values."""
    if df.empty or len(df) < 20:
        return "No Data Available"
    last_row = df.iloc[-1]
    if last_row["ATR"] > 1.5:  # High volatility
        return {"RSI_Threshold": 85, "ADX_Threshold": 35}
    else:  # Normal volatility
        return {"RSI_Threshold": 80, "ADX_Threshold": 30}


# ---- Predictive Model (Linear Regression) ----
def predict_price(df):
    """Predicts the next closing price using a simple Linear Regression model."""
    if df.empty or len(df) < 20:
        return None
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values
    model = LinearRegression()
    model.fit(X, y)
    next_time_step = len(df) + 1
    predicted_price = model.predict([[next_time_step]])
    return predicted_price[0]

# ---- Spread Widening Detection ----
historical_spreads = []

def detect_spread_widening(bid, ask):
    """Detects abnormal widening of bid-ask spreads."""
    global historical_spreads
    current_spread = ask - bid
    historical_spreads.append(current_spread)
    if len(historical_spreads) > 100:
        historical_spreads.pop(0)
    avg_spread = np.mean(historical_spreads)
    widening_threshold = avg_spread * 2  # If spread doubles
    if current_spread > widening_threshold:
        return "High Spread Widening Detected"
    return "Normal Spread"

def detect_price_spikes(df, threshold=3.0):
    """Detects extreme price spikes using a Z-score."""
    if df.empty or len(df) < 30:
        return "Insufficient Data"

    try:
        df["Price_Change"] = df["Close"].pct_change()
        df["Z_Score"] = (df["Price_Change"] - df["Price_Change"].mean()) / df["Price_Change"].std()

        if abs(df["Z_Score"].iloc[-1]) > threshold:
            return "Extreme Price Spike Detected"
        return "No Price Anomaly"

    except Exception as e:
        logging.error(f"Error detecting price anomalies: {str(e)}")
        return "Error"


def detect_liquidity_anomaly(df):
    """Detects a sudden drop in trading volume."""
    if df.empty or "Volume" not in df.columns or len(df) < 30:
        return "Insufficient Data"

    try:
        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
        last_volume = df["Volume"].iloc[-1]
        avg_volume = df["Volume_MA"].iloc[-1]

        if last_volume < avg_volume * 0.5:  # If volume is 50% lower than normal
            return "Liquidity Drop Detected"
        return "Normal Liquidity"

    except Exception as e:
        logging.error(f"Error detecting liquidity anomalies: {str(e)}")
        return "Error"

def detect_anomalies(df, bid_price, ask_price):
    """Aggregates all anomaly detection methods into one function."""

    anomalies = {
        "Volatility Clustering": detect_volatility_clustering(df),
        "Price Spike": detect_price_spikes(df),
        "Liquidity Drop": detect_liquidity_anomaly(df),
        "Spread Widening": detect_spread_widening(bid_price, ask_price),
    }

    normal_conditions = {
        "Volatility Clustering": "Normal Volatility",
        "Price Spike": "No Price Anomaly",
        "Liquidity Drop": "Normal Liquidity",
        "Spread Widening": "Normal Spread",
    }

    # ✅ Detect any anomalies by checking if the values differ from normal conditions
    detected_anomalies = [key for key, value in anomalies.items() if value != normal_conditions.get(key, "Normal")]

    if detected_anomalies:
        anomaly_msg = f"⚠️ Anomalies Detected: {', '.join(detected_anomalies)}"
        logging.warning(anomaly_msg)
        return anomaly_msg

    return "No Anomalies Detected"

from arch import arch_model

def detect_volatility_clustering(df):
    """Detects persistent high or low volatility using a GARCH model."""
    if df.empty or len(df) < 30:
        return "Insufficient Data"

    try:
        df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        returns = df["Log_Returns"].dropna()

        if len(returns) < 30:  # Ensure we have enough data points
            return "Insufficient Data for GARCH"

        # Avoid numerical instability
        if returns.std() < 1e-5:
            returns = returns * 1000

        # ✅ Corrected GARCH Model
        model = arch_model(returns, vol="Garch", p=1, q=1, rescale=True)
        fitted_model = model.fit(disp="off")

        forecast_volatility = fitted_model.conditional_volatility.iloc[-1]

        threshold_high = returns.std() * 1.5
        threshold_low = returns.std() * 0.75

        if forecast_volatility > threshold_high:
            return "High Volatility Cluster Detected"
        elif forecast_volatility < threshold_low:
            return "Low Volatility Cluster Detected"
        return "Normal Volatility"

    except Exception as e:
        logging.error(f"GARCH Model Failed: {str(e)}")
        return "Volatility Analysis Failed"




def momentum_mean_reversion(df):
    """Momentum Trading with Mean Reversion Strategy."""
    if df.empty or len(df) < 20:
        return "No Data Available"

    last_row = df.iloc[-1]
    atr = last_row["ATR"]

    # Adaptive thresholds based on market volatility (ATR)
    rsi_threshold = 85 if atr > 1.5 else 80
    adx_threshold = 35 if atr > 1.5 else 30

    # Identify overall trend
    if last_row["EMA_50"] > last_row["EMA_200"] and last_row["ADX"] > adx_threshold:
        trend = "UP"
    elif last_row["EMA_50"] < last_row["EMA_200"] and last_row["ADX"] > adx_threshold:
        trend = "DOWN"
    else:
        return "No Strong Trend"

    # Mean Reversion Entry Signals
    if trend == "UP" and last_row["RSI"] < (100 - rsi_threshold) and last_row["DI+"] > last_row["DI-"]:
        return "Buy Pullback"
    elif trend == "DOWN" and last_row["RSI"] > rsi_threshold and last_row["DI+"] < last_row["DI-"]:
        return "Sell Rally"

    return "No Trade"


# ---- Breakout Strategy (Bollinger Bands & ATR) ----
def breakout_strategy(df):
    """Detects price breakouts using Bollinger Bands & ATR."""
    if df.empty or len(df) < 20:
        return "No Data Available"

    last_row = df.iloc[-1]

    # ✅ Use Correct Labels for Bollinger Bands
    if last_row["Close"] > last_row["BB_Upper"] and last_row["ATR"] > 0.001:
        return "Bullish Breakout"
    elif last_row["Close"] < last_row["BB_Lower"] and last_row["ATR"] > 0.001:
        return "Bearish Breakout"

    return "No Breakout"

def predict_price(df):
    """Predicts the next closing price using Linear Regression."""
    if df.empty or len(df) < 20:
        return None

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values

    model = LinearRegression()  # ✅ Now it is properly defined
    model.fit(X, y)

    next_time_step = len(df) + 1
    predicted_price = model.predict([[next_time_step]])

    return predicted_price[0]

def classify_market_condition(df):
    """Classifies market conditions based on volatility and trend indicators."""
    if df.empty or len(df) < 20:
        return "Unknown"

    last_row = df.iloc[-1]

    # ✅ Ensure all required indicators exist
    required_columns = ["ATR", "ADX", "RSI", "BB_Upper", "BB_Lower"]
    for col in required_columns:
        if col not in df.columns or pd.isna(last_row[col]):
            return "Unknown"

    # ✅ Check for High/Low Volatility
    if last_row["ATR"] > df["ATR"].mean() * 1.5:
        return "High Volatility"
    elif last_row["ATR"] < df["ATR"].mean() * 0.75:
        return "Low Volatility"

    # ✅ Identify Trending vs. Range-Bound Markets
    if last_row["ADX"] > 25:  # ADX > 25 = Strong Trend
        if last_row["RSI"] > 70:
            return "Overbought Trend"
        elif last_row["RSI"] < 30:
            return "Oversold Trend"
        return "Strong Trend"
    else:
        return "Sideways Market"  # ADX < 25 = No strong trend

    return "Unknown"
def real_time_market_analysis():
    """Runs continuous market analysis for multiple instruments."""
    while True:
        price_data = fetch_real_time_prices()
        if not price_data:
            logging.warning("⚠️ Skipping analysis due to missing market data.")
            time.sleep(30)
            continue

        for instrument, (bid_price, ask_price) in price_data.items():
            try:
                logging.info(f"📊 Processing {instrument}...")

                # ✅ Fetch Data & Compute Indicators
                df = fetch_historical_data(instrument)
                df = compute_indicators(df)

                # ✅ Confirm All Required Indicators Exist
                required_columns = ["MACD", "MACD_Signal", "RSI", "ADX", "EMA_20", "EMA_50", "BB_Upper", "BB_Lower","ROC"]
                missing_columns = [col for col in required_columns if col not in df.columns or df[col].isna().all()]

                if missing_columns:
                    logging.warning(f"⚠️ {instrument}: Missing indicators: {missing_columns}. Skipping analysis.")
                    print(f"⚠️ {instrument}: Skipping analysis due to missing indicators: {missing_columns}.")
                    continue

                # ✅ Generate Trading Signals
                trend_signal = trend_following_strategy(df)
                momentum_signal = momentum_strategy(df)
                macd_signal = macd_rsi_strategy(df)
                pinbar_signal = pinbar_strategy(df)
                adaptive_levels = adaptive_thresholds(df)
                breakout_signal = breakout_strategy(df)

                # ✅ Handle Errors in Prediction Model
                try:
                    predicted_price = predict_price(df) if "Close" in df.columns else None
                except Exception as e:
                    logging.error(f"⚠️ Prediction error for {instrument}: {str(e)}")
                    predicted_price = None

                # ✅ Detect Anomalies
                anomaly_detection = detect_anomalies(df, bid_price, ask_price)

                # ✅ Ensure classify_market_condition() is properly handled
                try:
                    market_condition = classify_market_condition(df)
                except NameError:
                    logging.error("❌ classify_market_condition() is not defined!")
                    market_condition = "Unknown"

                # ✅ Define Severe Anomalies That Justify Avoiding Trading
                severe_anomalies = ["Extreme Price Spike", "High Spread Widening"]
                detected_anomalies = anomaly_detection.replace("⚠️ Anomalies Detected: ", "").split(
                    ", ") if anomaly_detection else []

                # ✅ Avoid Trading ONLY if severe anomalies exist
                if any(anomaly in severe_anomalies for anomaly in detected_anomalies):
                    strategy = "Safe Default Strategy (Avoid Trading)"
                else:
                    # ✅ Prioritize Momentum-Based or Trend Strategies If Present
                    if trend_signal.startswith("Trend Following"):
                        strategy = "Trend Following: EMA Crossover + ADX Confirmation"
                    elif momentum_signal.startswith("Momentum: Strong"):
                        strategy = "Momentum-Based Trading Strategy"
                    elif breakout_signal.startswith("Bullish Breakout") or breakout_signal.startswith(
                            "Bearish Breakout"):
                        strategy = "Breakout Strategy (Bollinger Bands & ATR)"
                    elif pinbar_signal.startswith("Pinbar Reversal"):
                        strategy = "Pinbar Reversal Strategy"
                    elif market_condition == "Overbought Trend":
                        strategy = "Mean Reversion Strategy"
                    elif market_condition == "Oversold Trend":
                        strategy = "Mean Reversion Strategy"
                    elif market_condition == "Sideways Market":
                        strategy = "Range Trading Strategy (Scalping)"
                    elif market_condition == "High Volatility":
                        strategy = "High-Volatility Adaptive Strategy"
                    elif market_condition == "Low Volatility":
                        strategy = "Low-Volatility Mean Reversion Strategy"
                    else:
                        strategy = "Unknown Strategy"

                if any(anomaly in severe_anomalies for anomaly in detected_anomalies):
                    strategy = "Safe Default Strategy (Avoid Trading)"
                else:
                    # ✅ Determine Best Strategy Based on Market Conditions & Signals
                    if market_condition == "Strong Trend":
                        if trend_signal.startswith("Trend Following"):
                            strategy = "Trend Following: EMA Crossover + ADX Confirmation"
                        elif macd_signal.startswith("MACD + RSI"):
                            strategy = "Momentum-Based MACD & RSI Strategy"
                        elif breakout_signal.startswith("Bullish Breakout") or breakout_signal.startswith(
                                "Bearish Breakout"):
                            strategy = "Breakout Strategy (Bollinger Bands & ATR)"
                        else:
                            strategy = "Momentum-Based Trading Strategy"

                    elif market_condition == "Sideways Market":
                        if pinbar_signal.startswith("Pinbar Reversal"):
                            strategy = "Pinbar Reversal Strategy"
                        elif momentum_signal.startswith("Momentum: Strong"):
                            strategy = "Momentum-Based Trading Strategy"
                        else:
                            strategy = "Range Trading Strategy (Scalping)"

                    elif market_condition == "Overbought Trend":
                        strategy = "Mean Reversion Strategy"

                    elif market_condition == "Oversold Trend":
                        strategy = "Mean Reversion Strategy"

                    elif market_condition == "High Volatility":
                        if momentum_signal.startswith("Momentum: Strong") or breakout_signal.startswith("Breakout"):
                            strategy = "High-Volatility Momentum & Breakout Strategy"
                        else:
                            strategy = "Volatility-Adaptive Strategy"

                    elif market_condition == "Low Volatility":
                        strategy = "Low-Volatility Mean Reversion Strategy"

                    else:
                        strategy = "Unknown Strategy"

                # ✅ Logging & Output
                log_message = (
                    f"\n📈 ✅ Instrument: {instrument}\n"
                    f"🕒 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"💲 Bid Price: {bid_price:.5f}, Ask Price: {ask_price:.5f}\n"
                    f"Predicted Next Price: {f'{predicted_price:.5f}' if predicted_price is not None else 'N/A'}\n"


                    f"🌍 Market Condition: {market_condition}\n"
                    f"⚠️ Anomaly Detection: {anomaly_detection if anomaly_detection else 'None'}\n"
                    f"📊 Trend Signal: {trend_signal}\n"
                    f"⚡ Momentum Signal: {momentum_signal}\n"
                    f"📈 MACD + RSI Signal: {macd_signal}\n"
                    f"🔄 Pinbar Reversal Signal: {pinbar_signal}\n"
                    f"💰 Spread Widening: {detect_spread_widening(bid_price, ask_price)}\n"
                    f"📊 Volatility Clustering: {detect_volatility_clustering(df)}\n"
                    f"📢 Breakout Detection: {breakout_signal}\n"
                    f"🎯 Recommended Strategy: {strategy}\n"
                    "--------------------------------------------------"
                )

                logging.info(log_message)
                print(log_message)

            except Exception as e:
                logging.error(f"❌ Error processing {instrument}: {str(e)}")

        time.sleep(30)  # ⏳ Adjust timing as needed

# ---- Run Analysis ----
real_time_market_analysis()


