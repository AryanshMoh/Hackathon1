import requests
import pandas as pd
import datetime

def fetch_comed_prices():
    url = "https://hourlypricing.comed.com/api?type=5minutefeed"
    response = requests.get(url)
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['millisUTC'].astype(float) / 1000, unit='s')
    df['hour'] = df['timestamp'].dt.floor('H')
    df['price'] = df['price'].astype(float)

    # Average price per hour (¢/kWh)
    hourly = df.groupby('hour')['price'].mean().reset_index()
    hourly.rename(columns={'price': 'price_cents'}, inplace=True)

    # Convert cents → dollars
    hourly['price'] = hourly['price_cents'] / 100
    hourly.drop(columns=['price_cents'], inplace=True)

    # Save to CSV
    hourly[['price']].to_csv("data/prices.csv", index=False)
    print("✅ Fetched and saved hourly prices to data/prices.csv")
    print(hourly.head())

if __name__ == "__main__":
    fetch_comed_prices()

