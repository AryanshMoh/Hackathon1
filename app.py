import streamlit as st
import pandas as pd
from stable_baselines3 import PPO
from train_agent import train_agent, run_agent
from fetch_live_prices import fetch_comed_prices
from utils.display_utils import format_schedule_readable
from utils.appliance_data import appliance_defaults


# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="SmartEnergy Optimizer", layout="wide")
st.title("SmartEnergy Optimizer")
st.write("Plan your appliance usage intelligently using **ComEd’s day-ahead electricity prices** and AI optimization.")

# -------------------------------
# Fetch latest ComEd prices automatically
# -------------------------------
st.subheader("💲 Latest Energy Prices")

with st.spinner("Fetching latest ComEd day-ahead prices..."):
    df_prices = fetch_comed_prices()

if df_prices is not None and not df_prices.empty:
    st.line_chart(df_prices["price"], height=200)
    st.caption("Day-ahead electricity prices ($/kWh)")
else:
    st.error("⚠️ Could not fetch ComEd prices.")

prices = df_prices["price"].values if df_prices is not None else []

# -------------------------------
# Appliance Selection
# -------------------------------
st.subheader("🏠 Select Your Appliances")

st.info("Choose your appliances and adjust power or duration as needed. Defaults are based on realistic average usage.")

selected_appliances = []
num_appliances = st.slider("How many appliances to optimize?", 1, 10, 3)

for i in range(num_appliances):
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.selectbox(f"Appliance {i+1}", list(appliance_defaults.keys()), key=f"appliance_{i}")
    with col2:
        default_power = appliance_defaults[name]
        power = st.number_input(
            f"{name} Power (kWh)",
            min_value=0.0, max_value=10.0,
            value=float(default_power),
            step=0.1, format="%.2f",
            key=f"power_{i}"
        )
        st.caption(f"💡 Typical consumption: {default_power:.2f} kWh")
    with col3:
        duration = st.number_input(f"{name} Duration (hours)", min_value=1, max_value=8, value=2, key=f"duration_{i}")
    selected_appliances.append({"name": name, "power": power, "duration": duration})

appliances = selected_appliances
st.write("### ✅ Selected Appliances")
st.dataframe(appliances)

# -------------------------------
# Restriction Input (Sleep / No-use Hours)
# -------------------------------
st.subheader("😴 Time Restrictions")
sleep_start = st.number_input("Sleep Start Hour (0–23)", min_value=0, max_value=23, value=0)
sleep_end = st.number_input("Sleep End Hour (0–23)", min_value=0, max_value=23, value=8)

restricted_hours = list(range(sleep_start, sleep_end)) if sleep_end > sleep_start else []

# -------------------------------
# AI Optimization (Train + Run)
# -------------------------------
st.subheader("🤖 Smart Optimization")

if st.button("Run SmartEnergy AI Optimization"):
    with st.spinner("Training AI agent and optimizing schedule..."):
        model = train_agent(prices, appliances, restricted_hours)
        schedule = run_agent(model, prices, appliances, restricted_hours)

    readable = format_schedule_readable(schedule, appliances)
    st.success("✅ AI-Optimized Schedule:")
    st.json(readable)

    # --- Calculate cost savings ---
    total_cost_optimized = sum(
        prices[h] * next(a['power'] for a in appliances if a['name'] == name)
        for name, hours in schedule.items() if isinstance(hours, list)
        for h in hours
    )
    total_cost_peak = sum(
        max(prices) * next(a['power'] for a in appliances if a['name'] == name) * a['duration']
        for a in appliances
    )

    savings = total_cost_peak - total_cost_optimized
    st.metric("💰 Estimated Daily Savings", f"${savings:.2f}")

else:
    st.info("Press the button above to let the AI optimize your schedule based on ComEd’s upcoming prices.")
