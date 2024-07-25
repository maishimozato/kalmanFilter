import numpy as np

# Synthetic data for process noise estimation
actual_states = np.array([0.95, 0.90, 0.85, 0.80])  # Actual SOC values
predicted_states = np.array([0.94, 0.89, 0.83, 0.78])  # Predicted SOC values

# Compute deviations
deviations = actual_states - predicted_states

# Compute variance (Sigma_w for a single state dimension)
Sigma_w = np.var(deviations)

# For a multi-dimensional state, you would calculate the covariance matrix similarly
print("Estimated Sigma_w:", Sigma_w)
