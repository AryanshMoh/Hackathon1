import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EnergyEnvWithPreferences(gym.Env):
    """
    RL environment that balances cost optimization with user comfort preferences.
    """

    def __init__(self, prices, appliances, restricted_hours=None, preferences=None):
        super(EnergyEnvWithPreferences, self).__init__()
        self.prices = np.array(prices)
        self.appliances = appliances
        self.restricted_hours = restricted_hours or []
        self.preferences = preferences or {}  # User comfort preferences

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

    def _get_comfort_penalty(self, appliance_name, hour):
        """Calculate comfort penalty for running appliance at this hour"""
        if appliance_name not in self.preferences:
            return 0.0
        
        pref = self.preferences[appliance_name]
        penalty = 0.0
        
        # Penalty for avoided hours
        if hour in pref.get('avoid_hours', []):
            penalty += pref.get('avoid_penalty', 2.0)
        
        # Bonus for preferred hours (negative penalty)
        if hour in pref.get('preferred_hours', []):
            penalty -= pref.get('preferred_bonus', 1.0)
        
        return penalty

    def step(self, action):
        done = False
        reward = 0.0

        # If restricted hour → heavy penalize any attempted usage
        if self.current_hour in self.restricted_hours:
            reward -= 10.0 * np.sum(action)
            self.current_hour += 1
            if self.current_hour >= self.num_hours:
                done = True
            return self._get_obs(), float(reward), done, False, {}

        # Compute energy cost and comfort penalties
        total_cost = 0.0
        total_comfort_penalty = 0.0
        active_appliances = 0
        
        for i, a in enumerate(self.appliances):
            if action[i] == 1 and self.remaining_durations[a["name"]] > 0:
                active_appliances += 1
                
                # Energy cost
                cost = a["power"] * self.prices[self.current_hour]
                total_cost += cost
                
                # Comfort penalty
                comfort_penalty = self._get_comfort_penalty(a["name"], self.current_hour)
                total_comfort_penalty += comfort_penalty
                
                # Decrement remaining duration
                self.remaining_durations[a["name"]] -= 1

        # Reward: negative cost and comfort penalty
        reward -= total_cost
        reward -= total_comfort_penalty

        # Penalty for too many concurrent appliances (realistic load)
        if active_appliances > 2:
            reward -= 0.5 * (active_appliances - 2)

        self.current_hour += 1
        done = self.current_hour >= self.num_hours or all(v <= 0 for v in self.remaining_durations.values())

        # BIG PENALTY at end if appliances not fully scheduled
        if done:
            for name, remaining in self.remaining_durations.items():
                if remaining > 0:
                    reward -= 50.0 * remaining  # MASSIVE penalty for unscheduled hours - this should never happen

        return self._get_obs(), float(reward), done, False, {}