import streamlit as st
import pandas as pd
from optimizer import optimize_schedule
from fetch_live_prices import fetch_comed_prices
from stable_baselines3 import PPO
from energy_env import EnergyEnv
from train_agent import train_agent, run_agent
import seaborn as sns
import matplotlib.pyplot as plt


st.title("SmartEnergy Optimizer")
st.write("Plan your appliance usage to save money using **real-time electricity prices**!")

# -------------------------------
# Default appliance setup
# -------------------------------
default_appliances = [
    {"name": "Washer", "power": 0.5, "duration": 2},
    {"name": "Dryer", "power": 1.0, "duration": 1},
    {"name": "Dishwasher", "power": 1.2, "duration": 2}
]

st.write("### Example Appliances")
st.dataframe(default_appliances)

# -------------------------------
# Fetch live price data
# -------------------------------
st.subheader("Live Energy Prices")
if st.button("Fetch Live ComEd Prices"):
    fetch_comed_prices()
    st.success("Updated prices from ComEd live feed!")

try:
    df_prices = pd.read_csv("data/prices.csv")
    st.line_chart(df_prices["price"], height=200)
    st.caption("Current hourly energy prices ($/kWh)")
except FileNotFoundError:
    st.info("Live data not fetched yet — click the button above to load prices.")
    df_prices = None

# -------------------------------
# Upload your own CSV
# -------------------------------
st.subheader("Or Upload Your Own Data")
uploaded_file = st.file_uploader(
    "Upload hourly price data (CSV with 'price' column)",
    type=["csv"]
)

if uploaded_file is not None:
    df_prices = pd.read_csv(uploaded_file)

# -------------------------------
# Run optimizer if data available
# -------------------------------
if df_prices is not None:
    prices = df_prices["price"].values

    st.subheader("Optimization Options")

    if st.button("Run Linear Optimization (Quick)"):
        schedule = optimize_schedule(uploaded_file or "data/prices.csv", default_appliances)
        st.success("Optimal Schedule (Linear Optimizer)")
        st.json(schedule)

    if st.button("Train SmartEnergy+ AI Agent"):
        with st.spinner("Training reinforcement learning agent..."):
            model = train_agent(prices, default_appliances)
            st.success("Training complete!")

    if st.button("Run AI Optimization"):
        try:
            model = PPO.load("models/energy_agent")
            schedule = run_agent(model, prices, default_appliances)
            st.success("🤖 AI-Optimized Schedule:")
            st.json(schedule)
        except FileNotFoundError:
            st.error("No trained model found. Please train the AI agent first.")
else:
    st.info("Please upload a price CSV or fetch live data to begin.")
