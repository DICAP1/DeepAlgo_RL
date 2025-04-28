

from stable_baselines3 import PPO
from multi_asset_framework.envs.trading_env import TradingEnv
from multi_asset_framework.utils import compute_indicators
from multi_asset_framework.strategies.ppo_signal_generator import compute_additional_features
from multi_asset_framework.oanda_api import OandaV20API
import pandas as pd
import numpy as np
from datetime import datetime

MODEL_PATH = f"multi_asset_framework/models/ppo_model_{datetime.now():%Y%m%d_%H%M%S}.zip"

def train_model():
    """Trains a PPO trading model on multiple assets and evaluates its performance."""
    api = OandaV20API()
    instruments = ["EUR_USD", "XAU_USD", "SPX500_USD"]
    all_data = []

    for instrument in instruments:
        print(f"\n📊 Fetching training data for {instrument}...")
        df = api.fetch_training_data(instrument, total_candles=3000)
        df["Asset"] = instrument
        df = compute_indicators(df)
        df = compute_additional_features(df)
        df = df.dropna()
        all_data.append(df)

    full_df = pd.concat(all_data).reset_index(drop=True)
    print(f"✅ Features in Training: {full_df.columns.tolist()}")

    env = TradingEnv(df=full_df)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)

    model.save(MODEL_PATH)
    print(f"\n✅ PPO model trained and saved to {MODEL_PATH}")

    evaluate_model(model, env)

def evaluate_model(model, env, episodes=5):
    """Evaluates the PPO model over several episodes."""
    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        print(f"Episode {episode + 1} Reward: {total_reward:.2f}")

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\n🚀 Average Reward: {avg_reward:.2f}, Std: {std_reward:.2f}")

if __name__ == "__main__":
    train_model()