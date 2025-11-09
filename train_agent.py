import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from energy_env import EnergyEnv


def train_agent(prices, appliances, restricted_hours):
    """
    Train the PPO reinforcement learning agent using the given price data and restricted hours.
    """
    env = EnergyEnv(prices, appliances, restricted_hours)
    check_env(env, warn=True)

    # ⭐ Improved hyperparameters for better learning
    model = PPO(
        "MlpPolicy", 
        env, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0
    )
    
    # ⭐ More timesteps for better learning
    model.learn(total_timesteps=50000)
    model.save("models/energy_agent")

    return model


def run_agent(model, prices, appliances, restricted_hours):
    """
    Run the trained model to generate an optimized schedule.
    """
    env = EnergyEnv(prices, appliances, restricted_hours)
    obs, _ = env.reset()
    done = False

    schedule = {a["name"]: [] for a in appliances}

    while not done:
        # Record the hour BEFORE stepping
        current_hour = env.current_hour
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        # Record scheduled hour for each appliance
        for i, a in enumerate(appliances):
            if action[i] == 1:
                schedule[a["name"]].append(current_hour)

    # Format hours into human-readable ranges
    readable_schedule = {}
    for name, hours in schedule.items():
        if not hours:
            readable_schedule[name] = "Not scheduled"
            continue

        hours = sorted(set(hours))
        ranges = []
        start = prev = hours[0]
        for h in hours[1:]:
            if h != prev + 1:
                ranges.append(f"{start}:00–{prev+1}:00")
                start = h
            prev = h
        ranges.append(f"{start}:00–{prev+1}:00")
        readable_schedule[name] = ", ".join(ranges)

    return readable_schedule