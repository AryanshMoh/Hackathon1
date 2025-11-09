import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EnergyEnv(gym.Env):
    """
    Custom reinforcement learning environment for SmartEnergy.
    The goal is to schedule appliances across 24+ hours to minimize total cost
    while respecting restricted (unavailable) hours.
    """

    def __init__(self, prices, appliances, restricted_hours=None):
        super(EnergyEnv, self).__init__()
        self.prices = np.array(prices)
        self.appliances = appliances
        self.restricted_hours = restricted_hours or []

        self.num_hours = len(prices)
        self.num_appliances = len(appliances)

        # Observation: [current_hour] + appliance status (on/off)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1 + self.num_appliances,), dtype=np.float32
        )

        # Action: binary decision per appliance (0 = off, 1 = on)
        self.action_space = spaces.MultiBinary(self.num_appliances)

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_hour = 0
        self.remaining_durations = {a["name"]: a["duration"] for a in self.appliances}
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        status = [1.0 if self.remaining_durations[a["name"]] > 0 else 0.0 for a in self.appliances]
        return np.array([self.current_hour / self.num_hours] + status, dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0.0

        # If restricted hour → penalize any attempted usage
        if self.current_hour in self.restricted_hours:
            reward -= 5 * np.sum(action)
            self.current_hour += 1
            if self.current_hour >= self.num_hours:
                done = True
            return self._get_obs(), float(reward), done, False, {}

        # Compute energy cost and progress
        total_cost = 0
        active_appliances = 0
        for i, a in enumerate(self.appliances):
            if action[i] == 1 and self.remaining_durations[a["name"]] > 0:
                active_appliances += 1
                total_cost += a["power"] * self.prices[self.current_hour]
                self.remaining_durations[a["name"]] -= 1

        # Reward: negative cost (we want to minimize it)
        reward -= total_cost

        # Penalty for too many concurrent appliances (realistic load)
        if active_appliances > 2:
            reward -= 0.5 * (active_appliances - 2)

        self.current_hour += 1
        done = self.current_hour >= self.num_hours or all(v <= 0 for v in self.remaining_durations.values())
        
        #BIG PENALTY at end if appliances not scheduled
        if done:
            for name, remaining in self.remaining_durations.items():
                if remaining > 0:
                    reward -= 10.0 * remaining  # Heavy penalty for unscheduled hours

        return self._get_obs(), float(reward), done, False, {}