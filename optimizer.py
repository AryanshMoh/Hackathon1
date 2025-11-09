import pandas as pd
import pulp
import numpy as np

def optimize_schedule_lp(prices, appliances, restricted_hours=None):
    """
    Linear programming optimizer - finds the absolute cheapest schedule.
    Guaranteed optimal but ignores user preferences/comfort.
    
    Args:
        prices: Array of hourly prices
        appliances: List of appliance dicts with name, power, duration
        restricted_hours: List of hour indices to avoid
    
    Returns:
        schedule: Dict mapping appliance names to list of hours
        total_cost: Total electricity cost
    """
    num_hours = len(prices)
    hour_indices = range(num_hours)
    restricted_hours = restricted_hours or []

    model = pulp.LpProblem("CostOptimization", pulp.LpMinimize)

    # Binary variable for each appliance-hour
    run = {
        (a['name'], h): pulp.LpVariable(f"{a['name']}_hour{h}", cat="Binary")
        for a in appliances for h in hour_indices
    }

    # Objective: minimize total cost
    model += pulp.lpSum(
        run[(a['name'], h)] * a['power'] * prices[h]
        for a in appliances for h in hour_indices
    )

    # Constraints: each appliance runs exactly for its duration
    for a in appliances:
        model += pulp.lpSum(run[(a['name'], h)] for h in hour_indices) == a['duration']

        # Cannot run in restricted hours
        for h in restricted_hours:
            if h in hour_indices:
                model += run[(a['name'], h)] == 0

    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extract schedule
    schedule = {}
    for a in appliances:
        hours_on = [h for h in hour_indices if pulp.value(run[(a['name'], h)]) == 1]
        schedule[a['name']] = hours_on if hours_on else []

    # Calculate total cost
    total_cost = sum(
        prices[h] * a['power']
        for a in appliances
        for h in schedule[a['name']]
    )

    return schedule, total_cost


def format_schedule_readable(schedule, appliances):
    """Format schedule into human-readable time ranges with better aesthetics"""
    readable = {}

    for name, hours in schedule.items():
        if not hours or len(hours) == 0:
            # This should never happen in normal operation
            # If it does, it indicates a scheduling failure
            readable[name] = "⚠️ Scheduling failed"
            continue

        hours = sorted(hours)
        ranges = []
        start = prev = hours[0]

        for h in hours[1:]:
            if h != prev + 1:
                # Format times nicely (e.g., "2:00 PM" instead of "14:00")
                ranges.append(format_time_range(start, prev + 1))
                start = h
            prev = h
        ranges.append(format_time_range(start, prev + 1))

        # Join with nice separator
        readable[name] = " | ".join(ranges)

    return readable


def format_time_range(start_hour, end_hour):
    """Format hour range into 12-hour format"""
    def hour_to_12h(h):
        if h == 0:
            return "12:00 AM"
        elif h < 12:
            return f"{h}:00 AM"
        elif h == 12:
            return "12:00 PM"
        else:
            return f"{h - 12}:00 PM"

    return f"{hour_to_12h(start_hour)} – {hour_to_12h(end_hour)}"