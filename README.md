# Kalman Filters: State Estimation for Lithium-Ion Batteries

Kalman Filters are essential tools for estimating the state of dynamic systems, particularly useful when dealing with noisy measurements and uncertain dynamics. This README provides an overview of Kalman Filters, focusing on their application to State of Charge (SoC) estimation in Lithium-Ion batteries.

pdf version:
[kalmanFilter.pdf](https://github.com/user-attachments/files/16104321/kalmanFilter.pdf)


## Overview

Kalman Filters operate on the principle of blending predictions from a system model with actual measurements to provide the best estimate of the system's state over time. This is particularly useful in scenarios where measurements are noisy and the dynamics of the system are not perfectly known.

### Key Concepts

#### State Estimation

Kalman Filters estimate the true state of a system, which is hidden, using noisy measurements. The filter iteratively updates its estimate based on predictions and measurements, adjusting for uncertainties.

#### Algorithm Steps

1. **Initialization:**
   - Initialize the state of the filter.
   - Establish an initial belief in the state.

2. **Predict:**
   - Use the system model to predict the state at the next time step.
   - Adjust the belief to account for prediction uncertainties.

3. **Update:**
   - Obtain a measurement and assess its accuracy.
   - Compute the residual between the estimated state and measurement.
   - Adjust the state estimate based on the measurement using the Kalman gain.

## Kalman Filter Algorithm

### Predict Step

#### State Propagation

The predict step involves propagating the state forward using the system model and adjusting for process noise.

$$ \hat{x}_k^- = A \hat{x}_{k-1} + B u_k $$
$$ P_k^- = A P_{k-1} A^T + Q_k $$

- $ \hat{x}_k^- $: Predicted state estimate at time $ k $
- $ A $: State transition matrix
- $ B $: Control input matrix
- $ u_k $: Control input at time $ k $
- $ P_k^- $: Predicted state covariance matrix
- $ Q_k $: Process noise covariance matrix

### Update Step

#### Measurement Incorporation

In the update step, the filter combines the predicted state with the actual measurement to refine the state estimate.

$$ K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^{-1} $$
$$ \hat{x}_k = \hat{x}_k^- + K_k (z_k - H_k \hat{x}_k^-) $$
$$ P_k = (I - K_k H_k) P_k^- $$

- K_k: Kalman gain
- H_k: Observation matrix
- R_k: Measurement noise covariance matrix
- z_k: Actual measurement at time k 

### Observation Model

The observation model predicts the expected measurement based on the current state estimate.

$$ z_k = H_k \hat{x}_k^- + \text{noise} $$

- **Noise**: Represents uncertainty in the measurement.

## Improving the Filter

Continuous improvement of the Kalman Filter involves adjusting parameters such as process noise covariance (Q_k) and measurement noise covariance (R_k) to better match the characteristics of the system being modeled.

### State Covariance Matrix

The state covariance matrix ($ P_k $) provides insight into the accuracy of the state estimate after incorporating measurement information.

$$ P_k = (I - K_k H_k) P_k^- $$

## Conclusion

Kalman Filters offer a robust method for state estimation in dynamic systems like Lithium-Ion batteries. By integrating predictions with measurements and adjusting for uncertainties, the filter provides an optimal estimate of the system's state over time.

For implementation details and practical examples, refer to the provided code.

