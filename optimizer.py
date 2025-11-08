import pandas as pd
import pulp

def optimize_schedule(prices_file, appliances):
    """
    Finds the cheapest hours to run each appliance based on hourly electricity prices
    and returns a clean, human-readable schedule.
    """

    # 1. Read hourly price data (e.g., 24 hours)
    prices = pd.read_csv(prices_file)
    num_hours = len(prices)
    hour_indices = range(num_hours)

    # 2. Define optimization problem
    model = pulp.LpProblem("SmartEnergyPlanner", pulp.LpMinimize)

    # 3. Binary variables: 1 if appliance runs during hour h
    run = {}
    for appliance in appliances:
        for hour in hour_indices:
            run[(appliance['name'], hour)] = pulp.LpVariable(f"{appliance['name']}_hour{hour}", cat="Binary")

    # 4. Objective: minimize total cost = sum(price * power * run)
    total_cost = [
        run[(appl['name'], h)] * appl['power'] * prices.loc[h, 'price']
        for appl in appliances
        for h in hour_indices
    ]
    model += pulp.lpSum(total_cost)

    # 5. Constraints: each appliance must run for its duration
    for appl in appliances:
        model += pulp.lpSum(run[(appl['name'], h)] for h in hour_indices) == appl['duration']

        # Optional: limit to allowed hours if provided
        if 'allowed_hours' in appl:
            for h in hour_indices:
                if h not in appl['allowed_hours']:
                    model += run[(appl['name'], h)] == 0

    # 6. Solve
    model.solve()

    # 7. Extract results
    schedule = {}
    for appl in appliances:
        hours_on = [h for h in hour_indices if pulp.value(run[(appl['name'], h)]) == 1]
        schedule[appl['name']] = hours_on

    # 8. Format for readability
    readable_schedule = {}
    for appl, hours in schedule.items():
        if not hours:
            readable_schedule[appl] = "Not scheduled"
            continue

        # Merge consecutive hours into ranges
        ranges = []
        start = prev = hours[0]
        for h in hours[1:]:
            if h == prev + 1:
                prev = h
            else:
                ranges.append((start, prev))
                start = prev = h
        ranges.append((start, prev))

        # Format ranges into readable strings
        readable = [f"{s}:00–{e+1}:00" if s != e else f"{s}:00–{s+1}:00" for s, e in ranges]
        readable_schedule[appl] = ", ".join(readable)

    return readable_schedule


# Example test run
if __name__ == "__main__":
    appliances = [
        {"name": "Washer", "power": 1.2, "duration": 2},
        {"name": "Dryer", "power": 1.5, "duration": 1},
        {"name": "Dishwasher", "power": 1.0, "duration": 2}
    ]

    result = optimize_schedule("data/prices.csv", appliances)
    print(result)
