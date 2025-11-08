import pandas as pd
import pulp

def optimize_schedule(prices_file, appliances):
    """
    Finds the cheapest hours to run each appliance based on hourly electricity prices.
    """

    # 1. Read the hourly price data (e.g., 24 hours)
    prices = pd.read_csv(prices_file)
    num_hours = len(prices)
    hour_indices = range(num_hours)

    # 2. Create a new optimization problem
    model = pulp.LpProblem("SmartEnergyPlanner", pulp.LpMinimize)

    # 3. Create binary variables:
    #    1 if the appliance runs during hour h, otherwise 0
    run = {}
    for appliance in appliances:
        for hour in hour_indices:
            var_name = f"{appliance['name']}_hour{hour}"
            run[(appliance['name'], hour)] = pulp.LpVariable(var_name, cat="Binary")

    # 4. Objective: minimize total cost = sum(price * power * run)
    total_cost = []
    for appliance in appliances:
        for hour in hour_indices:
            cost = run[(appliance['name'], hour)] * appliance['power'] * prices.loc[hour, 'price']
            total_cost.append(cost)
    model += pulp.lpSum(total_cost)

    # 5. Constraints: each appliance must run for its full duration
    for appliance in appliances:
        # Add constraint: sum of run hours must equal duration
        model += pulp.lpSum(run[(appliance['name'], hour)] for hour in hour_indices) == appliance['duration']

        # Optional: if allowed hours are given, disable other hours
        if 'allowed_hours' in appliance:
            for hour in hour_indices:
                if hour not in appliance['allowed_hours']:
                    model += run[(appliance['name'], hour)] == 0

    # 6. Solve the optimization
    model.solve()

    # 7. Extract results into a readable dictionary
    schedule = {}
    for appliance in appliances:
        hours_on = []
        for hour in hour_indices:
            if pulp.value(run[(appliance['name'], hour)]) == 1:
                hours_on.append(hour)
        schedule[appliance['name']] = hours_on

    return schedule


# Example test run
if __name__ == "__main__":
    appliances = [
        {"name": "Washer", "power": 1.2, "duration": 2},
        {"name": "Dryer", "power": 1.5, "duration": 1},
        {"name": "Dishwasher", "power": 1.0, "duration": 2}
    ]

    result = optimize_schedule("data/prices.csv", appliances)
    print(result)
