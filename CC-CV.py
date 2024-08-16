import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("NOTICE")

experiment = pybamm.Experiment([
    ("Discharge at 1C for 10 seconds", "Rest for 5 minutes", "Charge at 1C for 10 seconds", "Rest for 5 minutes"),
    # Decrease SOC by 10% using a constant current discharge (C/3)
    ("Discharge at C/3 for 5 minutes", "Rest for 5 minutes"),
    # Continue until SOC range of interest is covered
] * 10)

temperatures = [0, 10, 20, 30]

solutions = {}

def get_parameter_values(temp_c):
    # Convert Celsius to Kelvin
    temp_k = temp_c + 273.15

    # Create a model
    model = pybamm.lithium_ion.DFN()

    # Load default parameters
    parameter_values = pybamm.ParameterValues("Chen2020")
    
    # Example of updating parameters with temperature (adapt as needed)
    parameter_values.update({
        "Nominal cell capacity [A.h]": 5.8,
        "Lower voltage cut-off [V]": 2.5,
        "Upper voltage cut-off [V]": 4.2,
        "Electrolyte conductivity [S.m-1]": 1.0 / (1 + 0.01 * (temp_k - 298)),  # Example: decreasing conductivity with temperature
        "Negative electrode exchange-current density [A.m-2]": 1.0 * (1 + 0.005 * (temp_k - 298)),  # Example adjustment
    }, check_already_exists=False)
    
    return parameter_values, model

for temp_c in temperatures:

    print(f"Running simulation at {temp_c}°C")
    parameter_values, model = get_parameter_values(temp_c)

    sim = pybamm.Simulation(model, experiment=experiment, parameter_values=parameter_values)

    solution = sim.solve()
    solutions[temp_c] = solution
    
fig, axs = plt.subplots(len(temperatures), 1, figsize=(12, 8), sharex=True, sharey=True)

for i, (temp_c, solution) in enumerate(solutions.items()):
    axs[i].plot(solution.t, solution["Terminal voltage [V]"].entries)
    axs[i].set_title(f"Temperature = {temp_c}°C")
    axs[i].set_ylabel("Terminal Voltage [V]")
    axs[i].grid(True)

axs[-1].set_xlabel("Time [s]")

plt.tight_layout()
plt.show() 
    
#sim.plot(["Terminal voltage [V]", "Current [A]"])