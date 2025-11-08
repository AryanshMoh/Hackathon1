# train_agent.py
from stable_baselines3 import PPO
from energy_env import EnergyEnv
from utils.display_utils import format_schedule_readable
import numpy as np

def train_agent(prices, appliances):
    """
    Trains a PPO reinforcement learning agent on the custom energy environment.
    """
    env = EnergyEnv(prices, appliances)
    model = PPO("MlpPolicy", env, verbose=0, n_steps=256, batch_size=64)
    model.learn(total_timesteps=20_000)
    model.save("models/energy_agent")
    return model


def run_agent(model, prices, appliances):
    """
    Runs a trained RL agent on the environment to produce a readable schedule.
    """
    env = EnergyEnv(prices, appliances)
    obs, _ = env.reset()
    done = False

    # Store on-hours for each appliance
    raw_schedule = {a["name"]: [] for a in appliances}

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        # Record which appliances are ON during this hour
        current_hour = int(obs[0]) - 1
        if current_hour < 0:
            current_hour = 0

        for i, a in enumerate(appliances):
            if action[i] == 1:
                raw_schedule[a["name"]].append(current_hour)

    # Convert to human-readable format safely
    formatted_schedule = format_schedule_readable(raw_schedule, appliances)
    return formatted_schedule