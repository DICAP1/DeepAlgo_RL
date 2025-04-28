from datetime import datetime
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from multi_asset_framework.utils import compute_indicators
from multi_asset_framework.config.feature_config import FEATURE_COLUMNS

# 🔹 Load PPO model
from stable_baselines3 import PPO

def load_ppo_model(model_path):
    return PPO.load(model_path)


def compute_additional_features(df):
    """Adds engineered features to the DataFrame."""
    df['Pct_Change'] = df['Close'].pct_change()
    df['Delta_RSI'] = df['RSI'].diff()
    df['ZScore_Close_EMA50'] = (df['Close'] - df['EMA_50']) / df['Close'].rolling(window=20).std()
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    return df

def generate_trade_signal(df, asset_name=""):
    """Uses PPO model to generate trading signals with richer features."""
    df = compute_indicators(df)
    df = compute_additional_features(df)
    df = df.dropna()

    latest = df.iloc[-1]

    # ✅ Build observation from centralized feature list
    feature_vector = [latest.get(f, 0) for f in FEATURE_COLUMNS]
    obs = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

    # 🔹 PPO Prediction
    action, _ = ppo_model.predict(obs)
    logits = ppo_model.policy.forward(ppo_model.policy.obs_to_tensor(obs)[0])[0]
    confidence = float(logits[0][action])

    print(f"🤖 PPO Observation for {asset_name}: Shape {obs.shape}, Data: {obs}")
    print(f"🤖 PPO Model Signal for {asset_name}: {['HOLD', 'BUY', 'SELL'][action]} (Confidence: {confidence:.2f})")

    return (
        "BUY" if action == 1 else
        "SELL" if action == 2 else
        "HOLD",
        confidence
    )
