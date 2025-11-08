import pandas as pd

def load_appliance_power_dataset(csv_path_or_file):
    """
    Loads your appliance power dataset and converts it to the optimizer format:
    [
        {"name": "Washer", "power": 0.5, "duration": 2},
        ...
    ]
    """

    df = pd.read_csv(csv_path_or_file)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Normalize and auto-detect relevant columns
    name_col = next((c for c in df.columns if "appliance" in c), None)
    power_col = next((c for c in df.columns if "kw" in c), None)
    duration_col = next((c for c in df.columns if "hour" in c or "duration" in c), None)

    if not all([name_col, power_col, duration_col]):
        raise ValueError(f"Missing one or more required columns in file. Found: {df.columns.tolist()}")

    appliances = []
    for _, row in df.iterrows():
        try:
            appliances.append({
                "name": str(row[name_col]).strip(),
                "power": float(row[power_col]),
                "duration": int(round(float(row[duration_col]))),
            })
        except Exception as e:
            print(f"Skipping invalid row: {row.to_dict()}, error: {e}")
    return appliances
