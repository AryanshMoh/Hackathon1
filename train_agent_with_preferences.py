import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from energy_env_with_preferences import EnergyEnvWithPreferences


def train_agent_with_preferences(prices, appliances, restricted_hours, preferences):
    """
    Train RL agent that balances cost + user comfort preferences.
    """
    env = EnergyEnvWithPreferences(prices, appliances, restricted_hours, preferences)
    check_env(env, warn=True)

    # Improved hyperparameters
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
    
    # Train the model
    model.learn(total_timesteps=50000)
    model.save("models/energy_agent_preferences")

    return model


def run_agent_with_preferences(model, prices, appliances, restricted_hours, preferences):
    """
    Run trained model to generate preference-aware schedule.
    """
    env = EnergyEnvWithPreferences(prices, appliances, restricted_hours, preferences)
    obs, _ = env.reset()
    done = False

    schedule = {a["name"]: [] for a in appliances}

    while not done:
        # Record the hour and remaining durations BEFORE stepping
        current_hour = env.current_hour
        remaining_before_step = {a["name"]: env.remaining_durations[a["name"]] for a in appliances}
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        # Only record if appliance had remaining duration before the step
        for i, a in enumerate(appliances):
            if action[i] == 1 and remaining_before_step[a["name"]] > 0:
                schedule[a["name"]].append(current_hour)

    return schedule

def calculate_comfort_score(schedule, preferences):

    total_score = 0
    total_possible = 0

    for appliance_name, pref in preferences.items():
        if appliance_name not in schedule:
            continue

        hours = schedule[appliance_name]
        avoid_hours = pref.get("avoid_hours", [])
        preferred_hours = pref.get("preferred_hours", [])
        prefer_bonus = pref.get("preference_strength", 3)  # 1–5 scale

        # weightings
        avoid_penalty = 3 * prefer_bonus      # heavy penalty for violating avoids
        prefer_reward = 4 * prefer_bonus      # reward for following preferences
        neutral_reward = 0.5                  # small baseline
        completion_bonus = 2 * prefer_bonus   # extra if all required hours are satisfied

        score = 0
        for hour in hours:
            if hour in avoid_hours:
                score -= avoid_penalty
            elif hour in preferred_hours:
                score += prefer_reward
            else:
                score += neutral_reward

        # full satisfaction bonus (no avoids, all in preferred if possible)
        if all(h in preferred_hours for h in hours) and not any(h in avoid_hours for h in hours):
            score += completion_bonus

        total_score += max(score, 0)  # no negative totals
        total_possible += len(hours) * prefer_reward + completion_bonus

    # normalize to 0–10
    if total_possible == 0:
        return 0.0

    normalized = (total_score / total_possible) * 10
    return round(min(10, normalized), 2)
