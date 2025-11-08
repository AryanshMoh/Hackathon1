import streamlit as st
import pandas as pd
from optimizer import optimize_schedule
from fetch_live_prices import fetch_comed_prices

st.title("SmartEnergy Optimizer")
st.write("Plan your appliance usage to save money using **real-time electricity prices**!")

# --- Live data section ---
st.subheader("Live Energy Prices")
if st.button("Fetch Live ComEd Prices"):
    fetch_comed_prices()
    st.success("Updated prices from ComEd live feed!")

# Display latest prices if file exists
try:
    df = pd.read_csv("data/prices.csv")
    st.line_chart(df['price'], height=200)
    st.caption("Current hourly energy prices ($/kWh)")
except FileNotFoundError:
    st.info("Live data not fetched yet — click the button above to load prices.")

# --- Optimization section ---
st.subheader("Optimize Your Appliance Usage")

# Upload custom CSV (optional)
prices_file = st.file_uploader(
    "Or upload your own hourly price data (CSV with 'price' column)",
    type=["csv"]
)

# Example appliances
default_appliances = [
    {"name": "Washer", "power": 0.5, "duration": 2},
    {"name": "Dryer", "power": 1.0, "duration": 1},
    {"name": "Dishwasher", "power": 1.2, "duration": 2}
]

st.write("### Example appliances")
st.dataframe(default_appliances)

# Run optimizer
if prices_file is not None:
    st.write("Running optimization on uploaded data...")
    schedule = optimize_schedule(prices_file, default_appliances)
    st.success("Optimal Schedule:")
    st.json(schedule)
elif "data/prices.csv":
    st.write("Running optimization on latest live data...")
    schedule = optimize_schedule("data/prices.csv", default_appliances)
    st.success("Optimal Schedule:")
    st.json(schedule)
else:
    st.info("Please upload a CSV or fetch live data to start optimization.")
