import pandas as pd
import pulp

def optimize_schedule(prices_file, appliances, restricted_hours=None):
    """
    Finds the cheapest hours to run each appliance based on hourly electricity prices,
    while completely avoiding restricted hours.
    """
    prices = pd.read_csv(prices_file)
    num_hours = len(prices)
    hour_indices = range(num_hours)

    model = pulp.LpProblem("SmartEnergyPlanner", pulp.LpMinimize)

    # Binary variable for each appliance-hour
    run = {
        (a['name'], h): pulp.LpVariable(f"{a['name']}_hour{h}", cat="Binary")
        for a in appliances for h in hour_indices
    }

    # Objective: minimize total cost
    model += pulp.lpSum(run[(a['name'], h)] * a['power'] * prices.loc[h, 'price']
                        for a in appliances for h in hour_indices)

    # Constraints: each appliance runs exactly for its duration
    for a in appliances:
        model += pulp.lpSum(run[(a['name'], h)] for h in hour_indices) == a['duration']

        # HARD RESTRICTION: cannot run in restricted hours
        if restricted_hours:
            for h in restricted_hours:
                if h in hour_indices:
                    model += run[(a['name'], h)] == 0

    model.solve()

    # Format schedule
    schedule = {}
    for a in appliances:
        hours_on = [h for h in hour_indices if pulp.value(run[(a['name'], h)]) == 1]
        if not hours_on:
            schedule[a['name']] = "Not scheduled"
            continue

        # Merge consecutive hours
        ranges = []
        start = prev = hours_on[0]
        for h in hours_on[1:]:
            if h == prev + 1:
                prev = h
            else:
                ranges.append((start, prev))
                start = prev = h
        ranges.append((start, prev))

        readable = [f"{s}:00–{e+1}:00" for s, e in ranges]
        schedule[a['name']] = ", ".join(readable)

    return schedule
