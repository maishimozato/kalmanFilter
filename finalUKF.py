import numpy as np

# these will need to be calculated using the pulse discharge test
R0 = 1
R1 = 1
C1 = 1
R2 = 1
C2 = 1
Q = 1

tau1 = R1*C1
tau2 = R2*C2
Qnom = 5
deltaT = 1

A = np.array([1,0], [0, np.exp(-deltaT/(R1*C1))])
B = np.array([-deltaT/Qnom, R1*(1 - np.exp(-deltaT/tau1)), R2*(1 - np.exp(-deltaT/tau2))])


# template for ekf algorithm
def ekf(z_k_observation_vector, state_estimate_k_minus_1, P_k_minus_1):

    # state transition matrix 
    A_k = np.eye(3)  
    # identity matrix - no change in the state model from the previous state

    # define the process noise vector
    process_noise_v_k_minus_1 = np.array([0.01, 0.01, 0.003])

    # define the state model noise covariance matrix Q_k
    Q_k = np.eye(3)  
    # identity matrix - equal noise in all dimensions

    # define the measurement matrix H_k 
    H_k = np.eye(3)
    # identity matrix - measurements directly represent the state variables

    # define the measurement noise covariance matrix R_k 
    R_k = np.eye(3)
    # identity matrix - equal measurement noise in all dimensions

    # Predict Step
    state_estimate_k = A_k @ state_estimate_k_minus_1
    #calculate the predicted state using the state transition matrix and the previous state estimate
    
    P_k = A_k @ P_k_minus_1 @ A_k.T + Q_k
    #predict how much uncertainty there is in the state estimate using the process noise covariance matrix

    # Measurement Residual
    measurement_residual_y_k = z_k_observation_vector - H_k @ state_estimate_k

    # Innovation Covariance
    S_k = H_k @ P_k @ H_k.T + R_k

    # Kalman Gain
    K_k = P_k @ H_k.T @ np.linalg.pinv(S_k)

    # Update Step
    state_estimate_k = state_estimate_k + K_k @ measurement_residual_y_k
    P_k = (np.eye(3) - K_k @ H_k) @ P_k

    return state_estimate_k, P_k