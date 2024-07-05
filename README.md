# Kalman Filters: A Brief Overview

Kalman Filters are powerful tools for state estimation, particularly useful when dealing with noisy measurements and uncertain dynamics. They provide an optimal way to estimate the state of a system over time based on a series of measurements.

## Key Concepts

### State Estimation

Kalman Filters estimate the state of a system based on noisy data. The estimation is refined by combining predictions from a system model with actual measurements.

### Algorithm Steps

#### Initialization

- Initialize the state of the filter.
- Establish an initial belief in the state.

#### Predict

- Use the system model to predict the state at the next time step.
- Adjust the belief to account for prediction uncertainties.

#### Update

- Obtain a measurement and assess its accuracy.
- Compute the residual between the estimated state and measurement.
- Adjust the state estimate based on the measurement using the Kalman gain.

## Kalman Filter Algorithm

### Predict Step

#### State Propagation

$ \hat{x}_k^- = A \hat{x}_{k-1} + B u_k $

$ P_k^- = A P_{k-1} A^T + Q_k $

### Update Step

#### Measurement Incorporation

$ K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^{-1} $

$ \hat{x}_k = \hat{x}_k^- + K_k (z_k - H_k \hat{x}_k^-) $

$ P_k = (I - K_k H_k) P_k^- $

### Observation Model

The observation model predicts what measurements we should observe based on the predicted state $ \hat{x}_k^- $.

$ z_k = H_k \hat{x}_k^- + \text{noise} $

- **Noise**: Represents uncertainty in the measurement.
