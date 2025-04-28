import os
import sys

# 🧠 Add the core directory to Python's path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from multi_asset_framework.oanda_api import OandaV20API
from multi_asset_framework.utils import compute_indicators
import pandas as pd

# ✅ Initialize OANDA API
api = OandaV20API()

# ✅ Fetch candles for a symbol
symbol = "EUR_USD"
df = api.fetch_candles(symbol)

# ✅ Check if data is returned
if df.empty:
    print(f"❌ No data fetched for {symbol}.")
else:
    print(f"✅ Successfully fetched {len(df)} candles for {symbol}.")

    # ✅ Rename columns to match indicator function expectations
    df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low'}, inplace=True)

    # ✅ Compute indicators
    df = compute_indicators(df)

    # ✅ Show output
    print("\n📊 Indicator Columns Preview:")
    print(df[['Close', 'EMA_50', 'EMA_200', 'RSI', 'ATR', 'MACD_Histogram']].tail())
