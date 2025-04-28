import gymnasium as gym
import numpy as np
from gymnasium import spaces
from multi_asset_framework.config.feature_config import FEATURE_COLUMNS

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.positions = []

        self.total_steps = len(self.df)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: HOLD, 1: BUY, 2: SELL
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(FEATURE_COLUMNS),),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.current_step = np.random.randint(0, self.total_steps - 100)  # Randomize start for each episode
        self.positions = []

        observation = self._next_observation()
        info = {}

        return observation, info

    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([row.get(f, 0) for f in FEATURE_COLUMNS], dtype=np.float32)
        return obs

    def step(self, action):
        reward = self._calculate_reward(action)

        # Advance step
        self.current_step += 1

        terminated = self.current_step >= self.total_steps - 1
        truncated = False  # can set additional truncation logic if needed

        observation = self._next_observation() if not terminated else np.zeros(self.observation_space.shape)

        info = {}

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, action):
        if self.current_step == 0:
            return 0

        price_now = self.df.iloc[self.current_step]["Close"]
        price_prev = self.df.iloc[self.current_step - 1]["Close"]
        price_change = price_now - price_prev

        reward = 0
        if action == 1:  # BUY
            reward = price_change
        elif action == 2:  # SELL
            reward = -price_change
        else:  # HOLD
            reward = -0.01  # Small penalty to discourage inaction

        return reward

    def render(self, mode="human"):
        pass  # Optional: Implement for visualization purposes later
