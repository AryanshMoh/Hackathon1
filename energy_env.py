import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EnergyEnv(gym.Env):
    def __init__(self, prices, appliances):
        super(EnergyEnv, self).__init__()

        self.prices = np.array(prices)
        self.appliances = appliances
        self.num_appliances = len(appliances)
        self.num_hours = len(prices)

        # Observation: current hour + remaining durations
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(self.num_appliances + 1,), dtype=np.float32
        )

        # Action: which appliances to turn ON (0 or 1 for each)
        self.action_space = spaces.MultiBinary(self.num_appliances)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hour = 0
        self.remaining = np.array([a["duration"] for a in self.appliances])
        obs = np.concatenate(([self.hour], self.remaining))
        return obs, {}

    def step(self, action):
        cost = 0
        energy_used = 0

        for i, a in enumerate(self.appliances):
            if action[i] == 1 and self.remaining[i] > 0:
                cost += a["power"] * self.prices[self.hour]
                energy_used += a["power"]
                self.remaining[i] -= 1

        reward = -cost  # minimize total cost
        self.hour += 1
        done = self.hour >= self.num_hours or np.all(self.remaining <= 0)

        obs = np.concatenate(([self.hour], self.remaining))
        info = {"cost": cost, "energy": energy_used}
        return obs, reward, done, False, info
