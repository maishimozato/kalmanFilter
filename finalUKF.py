import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sococvRegression():
    sococv = pd.read_excel("/Users/maishimozato/Documents/uoft ece - 2nd year/research/SOCOCV.xlsx")
    SOC = sococv['SOC'].values
    OCV = sococv['OCV'].values
    
    #fit a polynomial of degree 11 to the data
    coeffs = np.polyfit(SOC, OCV, 11)
    SOCOCV = np.poly1d(coeffs)
    
    SOC_fit = np.linspace(SOC.min(), SOC.max(), 500)
    OCV_fit = SOCOCV(SOC_fit)
    
    SOC_sample = SOC[::120]
    OCV_sample = OCV[::120]
    
    plt.figure(figsize=(10,6))
    
    plt.scatter(SOC_sample, OCV_sample, label="Data", s=10, color='blue')
    plt.plot(SOC_sample, OCV_sample, color='blue', linewidth=0.5)
    
    plt.plot(SOC_fit, OCV_fit, color='red', linewidth=2, label='Polynomial Fit')
    plt.xlabel('SOC')
    plt.ylabel('OCV')
    plt.title('SOC vs OCV with Polynomial Fit')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return SOCOCV

sococv = sococvRegression()

#a function that relates soc to ocv
def OCV(soc):
    SOCOCV = sococvRegression()
    OCV_value = SOCOCV(soc)
    return OCV_value

#a function that relates ocv to the terminal voltage
def g(x,D,u):
    terminal_voltage = OCV(x[0,0]) - x[1,0] + D*u
    return terminal_voltage

#the function to predict the new state value based on current state and control input (current) 
def f_function(A, B, x, u):
    x_new = A@x + B*u
    return x_new

def ekf():
    
    #initial SOC
    SOC_init = 1 
    #state space x - parameter initialization
    x_init = np.array([SOC_init, 0, 0])

    SOC_estimations = []
    Vterminal_estimations = []
    Vterminal_error = []
    
    # these will need to be calculated using the pulse discharge test
    R0 = 1
    R1 = 1
    C1 = 1
    R2 = 1
    C2 = 1
    Q = 1

    tau1 = R1*C1
    tau2 = R2*C2
    #time intervals in seconds
    deltaT = 1
    #capacity in amp-seconds to match time intervals
    Qnom = 5.8 * 3600 

    A = np.array([1,0,0], 
                 [0, np.exp(-deltaT/(tau1)),0], 
                 [0,0, np.exp(-deltaT/(tau2))])
    
    B = np.array([-deltaT/Qnom, R1*(1 - np.exp(-deltaT/tau1)), R2*(1 - np.exp(-deltaT/tau2))])

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

def plot(estimations, intervals):
    estimations = np.array(estimations)
    intervals = np.array(intervals)
    
    if estimations.ndim != 1:
        raise ValueError("estimations must be a 1D array")
    if intervals.ndim != 1:
        raise ValueError("intervals must be a 1D array")
    
    plt.figure(figsize=(10,6))
    plt.plot(intervals, estimations, marker='o')