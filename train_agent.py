# train_agent.py
from stable_baselines3 import PPO
from energy_env import EnergyEnv

def train_agent(prices, appliances):
    env = EnergyEnv(prices, appliances)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
    model.save("models/energy_agent")
    return model

def run_agent(model, prices, appliances):
    env = EnergyEnv(prices, appliances)
    obs, _ = env.reset()
    done = False
    schedule = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        schedule.append(action.tolist())
        obs, _, done, _, _ = env.step(action)

    return schedule
