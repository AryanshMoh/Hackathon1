# utils/appliance_data.py
# Updated default hourly kWh values based on average appliance energy use
# Sources:
# - Silicon Valley Power Appliance Energy Use Chart: https://www.siliconvalleypower.com/residents/save-energy/appliance-energy-use-chart
# - EnergySage: https://www.energysage.com/electricity/house-watts/how-many-watts-does-a-refrigerator-use/
# - IGS Energy: https://www.igs.com/energy-resource-center/energy-101/how-much-electricity-do-my-home-appliances-use


appliance_defaults = {
    "Washing Machine": 0.30,
    "Dryer": 2.50,
    "Dishwasher": 1.50,
    "Oven": 2.30,
    "Television": 0.10,
    "Computer": 0.10,
    "Air Conditioner": 3.50,
    "Heater": 1.50,
    "Lighting": 0.02
}