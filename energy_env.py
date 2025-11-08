import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EnergyEnv(gym.Env):
    """
    A reinforcement learning environment for optimizing appliance schedules
    given hourly electricity prices. The agent learns to minimize total cost
    while ensuring each appliance runs for its required duration.
    """

    def __init__(self, prices, appliances):
        super(EnergyEnv, self).__init__()

        # Normalize prices to 0–1 range to keep reward scale consistent
        prices = np.array(prices, dtype=np.float32)
        self.prices = (np.array(prices) - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-6)


        self.appliances = appliances
        self.num_appliances = len(appliances)
        self.num_hours = len(prices)

        # Observation: current hour + remaining durations
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.num_hours, 24),
            shape=(self.num_appliances + 1,),
            dtype=np.float32
        )

        # Action: which appliances to turn ON (0 or 1 for each)
        self.action_space = spaces.MultiBinary(self.num_appliances)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hour = 0
        # Each appliance starts with its full required duration remaining
        self.remaining = np.array([a["duration"] for a in self.appliances], dtype=np.float32)
        self.total_cost = 0.0
        obs = np.concatenate(([self.hour], self.remaining))
        return obs, {}

    def step(self, action):
        """
        Executes one hour of scheduling decisions.
        Action = binary vector (1 if appliance is ON this hour).
        Reward = negative cost (lower cost → higher reward)
                 plus penalties for unfinished or overused appliances.
        """
        cost = 0.0
        penalty = 0.0

        # Calculate energy cost and apply penalties
        for i, a in enumerate(self.appliances):
            if action[i] == 1 and self.remaining[i] > 0:
                cost += a["power"] * self.prices[self.hour]
                self.remaining[i] -= 1
            elif action[i] == 1 and self.remaining[i] <= 0:
                # Penalize running after completion
                penalty += 2.0

        # --- Reward shaping ---
        reward = -cost

                # Penalize running after an appliance is finished
        overuse_penalty = 0.0
        for i, rem in enumerate(self.remaining):
            if rem <= 0 and action[i] == 1:
                overuse_penalty += 5.0  # stronger penalty

        reward -= overuse_penalty
        penalty += overuse_penalty


        # Strong penalty for unfinished appliances at the end of the day
        if self.hour == self.num_hours - 1:
            unfinished_penalty = np.sum(self.remaining) * 10.0
            reward -= unfinished_penalty
            penalty += unfinished_penalty

        # Encourage finishing before the day ends
        if self.hour > self.num_hours * 0.75:
            late_penalty = np.sum(self.remaining) * 2.0
            reward -= late_penalty
            penalty += late_penalty

        self.total_cost += cost
        self.hour += 1

        done = bool(self.hour >= self.num_hours or np.all(self.remaining <= 0))
        obs = np.concatenate(([self.hour], self.remaining))
        info = {"cost": cost, "penalty": penalty, "total_cost": self.total_cost}

        return obs, reward, done, False, info
