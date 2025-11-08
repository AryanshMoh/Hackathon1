import pandas as pd

def format_schedule_readable(schedule, appliances):
    """
    Convert schedule dictionary of appliance-hour mappings into a human-readable form.
    """
    readable = {}

    for appliance, hours in schedule.items():
        if not hours:
            readable[appliance] = "Not scheduled"
            continue

        readable_hours = []
        for h in hours:
            try:
                hour = int(float(h))  # safely cast string or float to int
                readable_hours.append(f"{hour:02d}:00–{hour+1:02d}:00")
            except Exception as e:
                print(f"Skipping invalid hour '{h}' for {appliance}: {e}")

        readable[appliance] = ", ".join(readable_hours) if readable_hours else "Not scheduled"

    return readable
