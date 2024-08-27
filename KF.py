import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_points = 50
total_time_seconds = 10800  # 3 hours in seconds

# Generate time values from 0 to total_time_seconds
time = np.linspace(0, total_time_seconds, num=num_points)

# Generate true SoC values (decreasing from 100% to 0%)
true_soc = np.linspace(100, 0, num=num_points)

# Generate initial estimated SoC values starting at 100%
initial_estimated_soc = 100
estimated_soc = np.copy(true_soc)

# Simulate some minimal noise and convergence
np.random.seed(0)  # For reproducibility
noise = np.random.normal(0, 1, size=num_points)  # Minimal Gaussian noise
# Introduce a gradual convergence towards true_soc
estimated_soc = true_soc + (initial_estimated_soc - true_soc) * np.exp(-0.001 * time) + noise

# Ensure the first value of both true and estimated SoC is 100%
true_soc[0] = 100
estimated_soc[0] = 100

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(time, true_soc, 'r--', label='True SoC')
plt.plot(time, estimated_soc, 'o-', label='Estimated SoC')
plt.xlabel('Time (seconds)')
plt.ylabel('SoC (%)')
plt.title('True vs Estimated State of Charge (SoC)')
plt.legend()
plt.grid(True)
plt.show()


