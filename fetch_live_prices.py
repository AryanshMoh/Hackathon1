# fetch_live_prices.py
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import pytz

def generate_sample_prices():
    """
    Generates realistic sample electricity prices for a 24-hour period.
    Prices follow typical patterns: low at night, high during peak hours (afternoon/evening).
    """
    now = datetime.now(pytz.timezone("America/Chicago"))
    
    # Generate 24 hours of prices (hourly)
    hours = []
    prices = []
    
    for i in range(24):
        hour_time = now + timedelta(hours=i)
        hours.append(hour_time.strftime("%I:%M %p"))
        
        # Price pattern: low at night (2-6 AM), peak during day (2-8 PM)
        hour_of_day = hour_time.hour
        
        if 2 <= hour_of_day < 6:  # Night - cheapest
            base_price = 0.03
        elif 6 <= hour_of_day < 9:  # Morning ramp-up
            base_price = 0.05
        elif 9 <= hour_of_day < 14:  # Mid-day
            base_price = 0.07
        elif 14 <= hour_of_day < 20:  # Peak evening
            base_price = 0.10
        elif 20 <= hour_of_day < 23:  # Evening decline
            base_price = 0.06
        else:  # Late night
            base_price = 0.04
        
        # Add some random variation (+/- 20%)
        price = base_price * (1 + np.random.uniform(-0.2, 0.2))
        prices.append(round(price, 4))
    
    df = pd.DataFrame({
        "time": hours,
        "price": prices
    })
    
    return df

def fetch_comed_prices():
    """
    Fetches current 5-minute ComEd prices and cleans them for display.
    Falls back to sample data if API is unavailable or returns no future data.
    """
    URL = "https://hourlypricing.comed.com/api?type=5minutefeed&format=json"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        r = requests.get(URL, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()

        df = pd.DataFrame(data)
        df["millisUTC"] = df["millisUTC"].astype(int)
        df["price"] = df["price"].astype(float) / 100  # convert to $/kWh
        df["datetime"] = pd.to_datetime(df["millisUTC"], unit="ms", utc=True)
        df["datetime"] = df["datetime"].dt.tz_convert("America/Chicago")

        # Filter to only entries from 'now' forward
        now = datetime.now(pytz.timezone("America/Chicago"))
        df = df[df["datetime"] >= now]

        # ⭐ NEW CHECK: If filtering removed all data, use sample data
        if df.empty:
            print("⚠️ API returned only historical data (nothing in the future)")
            print("📊 Using sample price data instead...")
            df = generate_sample_prices()
            os.makedirs("data", exist_ok=True)
            df.to_csv("data/prices.csv", index=False)
            print(f"✅ Generated {len(df)} hours of sample price data.")
            return df

        # Format timestamps to AM/PM display
        df["time"] = df["datetime"].dt.strftime("%I:%M %p")

        os.makedirs("data", exist_ok=True)
        df[["time", "price"]].to_csv("data/prices.csv", index=False)

        print(f"✅ Loaded {len(df)} 5-minute data points from now onward.")
        return df[["time", "price"]]

    except Exception as e:
        print(f"⚠️ Could not fetch live ComEd prices: {e}")
        print("📊 Using sample price data instead...")
        
        # Generate and save sample data
        df = generate_sample_prices()
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/prices.csv", index=False)
        
        print(f"✅ Generated {len(df)} hours of sample price data.")
        return df