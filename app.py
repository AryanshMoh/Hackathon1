import streamlit as st
import pandas as pd
from utils.display_utils import format_schedule_readable
from optimizer import optimize_schedule
from fetch_live_prices import fetch_comed_prices
from stable_baselines3 import PPO
from train_agent import train_agent, run_agent
from utils.appliance_data import appliance_defaults


# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(page_title="SmartEnergy Optimizer", layout="wide")
st.title("SmartEnergy Optimizer")
st.write("Plan your appliance usage to save money using **real-time electricity prices** or your own uploaded data!")

# -------------------------------
# Appliance Selection
# -------------------------------
st.subheader("🏠 Select Your Appliances")

st.info("Choose from common household appliances. You can adjust the power and duration as needed.")

selected_appliances = []
num_appliances = st.slider("How many appliances do you want to optimize?", 1, 10, 3)

for i in range(num_appliances):
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.selectbox(
            f"Appliance {i+1}",
            list(appliance_defaults.keys()),
            key=f"appliance_{i}"
        )
    with col2:
        default_power = appliance_defaults[name]
        power = st.number_input(
            f"{name} Power (kWh)", value=default_power, step=0.1, key=f"power_{i}"
        )
    with col3:
        duration = st.number_input(
            f"{name} Duration (hours)", min_value=1, max_value=8, value=2, key=f"duration_{i}"
        )

    selected_appliances.append({"name": name, "power": power, "duration": duration})

appliances = selected_appliances
st.write("### ✅ Selected Appliances")
st.dataframe(appliances)

# -------------------------------
# Energy Price Data
# -------------------------------
st.subheader("💲 Energy Prices")

if st.button("Fetch Live ComEd Prices"):
    fetch_comed_prices()
    st.success("✅ Updated prices from ComEd live feed!")

try:
    df_prices = pd.read_csv("data/prices.csv")
    st.line_chart(df_prices["price"], height=200)
    st.caption("Current hourly energy prices ($/kWh)")
except FileNotFoundError:
    df_prices = None
    st.info("No live price data found — fetch from ComEd or upload below.")

uploaded_file = st.file_uploader("Or upload your own hourly price data (CSV with 'price' column)", type=["csv"])
if uploaded_file is not None:
    df_prices = pd.read_csv(uploaded_file)

# -------------------------------
# Run Optimization
# -------------------------------
if df_prices is not None:
    prices = df_prices["price"].values
    st.subheader("Optimization Options")

    # --- Linear Optimization ---
    if st.button("Run Linear Optimization (Quick)"):
        schedule = optimize_schedule(uploaded_file or "data/prices.csv", appliances)
        readable = format_schedule_readable(schedule, appliances)
        st.success("Optimal Schedule (Linear Optimizer):")
        st.json(readable)

    # --- Train RL Agent ---
    if st.button("Train SmartEnergy+ AI Agent"):
        with st.spinner("Training reinforcement learning agent... this may take a few minutes"):
            model = train_agent(prices, appliances)
            st.success("Training complete! Model saved as 'models/energy_agent'.")

    # --- Run RL Optimization ---
    if st.button("Run AI Optimization"):
        try:
            model = PPO.load("models/energy_agent")
            schedule = run_agent(model, prices, appliances)
            st.success("AI-Optimized Schedule:")
            st.json(schedule)
        except FileNotFoundError:
            st.error("No trained model found. Please train the AI agent first.")
else:
    st.info("Please fetch or upload price data to begin optimization.")
