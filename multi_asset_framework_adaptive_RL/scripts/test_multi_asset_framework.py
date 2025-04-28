import sys
import os

# Add the 'core' directory to sys.path so Python can recognize multi_asset_framework
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from multi_asset_framework.utils import compute_indicators
from multi_asset_framework.strategies.ppo_signal_generator import generate_trade_signal
from multi_asset_framework.strategies.rule_based import apply_rule_based_strategy
from multi_asset_framework.strategies.portfolio_allocator import allocate_portfolio
from multi_asset_framework.trade_execution import execute_multi_asset_trades
from multi_asset_framework.oanda_api import OandaV20API


# Initialize API
api = OandaV20API()

# Define assets to trade
assets = ["EUR_USD", "XAU_USD", "SPX500_USD"]

# Store trade signals
trade_signals = {}

for asset in assets:
    print(f"\n📊 Processing asset: {asset}")

    # 🔹 Fetch and prepare market data
    df = api.fetch_candles(asset)
    df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low'}, inplace=True)

    # 🔹 Compute indicators
    df = compute_indicators(df)

    # 🔹 Rule-Based signal
    rule_signal = apply_rule_based_strategy(df)

    # 🔹 PPO Model signal and confidence
    ppo_signal, confidence = generate_trade_signal(df, asset)


    print(f"📏 Rule-Based Signal for {asset}: {rule_signal}")
    print(f"🤖 PPO Model Signal for {asset}: {ppo_signal} (Confidence: {confidence:.2f})")

    # 🔹 Decision Logic: PPO refines Rule-Based signal
    if rule_signal == "BUY" and ppo_signal == "SELL":
        final_signal = "HOLD"
        print(f"⚠️ {asset} — Rule: BUY, PPO: SELL → Final: HOLD (PPO override)")
    elif rule_signal == "SELL" and ppo_signal == "BUY":
        final_signal = "HOLD"
        print(f"⚠️ {asset} — Rule: SELL, PPO: BUY → Final: HOLD (PPO override)")
    else:
        final_signal = ppo_signal
        print(f"✅ {asset} — Rule: {rule_signal}, PPO: {ppo_signal} → Final: {final_signal} (PPO confirmed or solo)")

    # 🔹 Store decision
    trade_signals[asset] = final_signal

# 🔹 Capital Allocation & Trade Execution
allocation = allocate_portfolio(trade_signals, capital=10000)
execute_multi_asset_trades(trade_signals)

print("\n✅ Multi-Asset Trading Framework Tested Successfully!")
