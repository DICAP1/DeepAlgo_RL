import pandas as pd

def apply_rule_based_strategy(df):
    """Simple rule-based strategy"""
    if len(df) < 50:
        return "NO SIGNAL"

    latest = df.iloc[-1]

    if latest["EMA_50"] > latest["EMA_200"] and latest["RSI"] < 35:
        return "BUY"
    elif latest["EMA_50"] < latest["EMA_200"] and latest["RSI"] > 65:
        return "SELL"
    return "HOLD"
