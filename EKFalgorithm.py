import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as s 
import seaborn as sns

from scipy.interpolate import interp1d

def sococvRegression():
    """
    - performs polynomial regression to model the relationship between soc and ocv
    - reads soc and ocv data from excel file, fits a polynomial of degree 11 to this data

    Returns:
        SOCOCV (np.ndarray) : array of polynomial coeffs fitted to soc-ocv data
        SOC (np.ndarray) : array of soc values from data
        OCV (np.ndarray) : array of ocv values corresponding to the soc values
    """
    sococv = pd.read_excel("/Users/maishimozato/Documents/uoftEce2ndYear/research/dataCollection/SOCOCV.xlsx")
    SOC = sococv['SOC'].values
    OCV = sococv['OCV'].values
    SOCOCV = np.polyfit(SOC, OCV, 11)
    
    return SOCOCV, SOC, OCV


def plotSOCOCV(SOC, OCV, SOCOCV):
    """
    - plots the soc-ocv polynomial fit curve
    - generates a set of evenly spaced values between the min and max values of soc data
    - compute corresponding ocv values
    """
    
    SOCOCV_func = np.poly1d(SOCOCV)
    SOC_fit = np.linspace(SOC.min(), SOC.max(), 500)
    OCV_fit = SOCOCV_func(SOC_fit)
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,6))
    
    plt.scatter(SOC, OCV, label="Original Data", s=15, color='blue', alpha=0.7)
    plt.plot(SOC_fit, OCV_fit, color='red', linewidth=2.5, linestyle='--', label='Polynomial Fit')
    
    plt.xlabel('State of Charge (SOC)', fontsize=14)
    plt.ylabel('Open Circuit Voltage (OCV)', fontsize=14)
    plt.title('SOC vs OCV with Polynomial Fit from C/5 OCV discharge test', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.show()
    
    return


def f_function(A, B, x, u):
    """
    - predicts the new state value in a state-space model
    - uses the current state and control input

    Parameters:
        A (np.ndarray): state transition matrix (shape: [n, n]).
        B (np.ndarray): control input matrix (shape: [n, m]).
        x (np.ndarray): current state vector (shape: [n, 1]).
        u (np.ndarray): control input vector (shape: [m, 1]).
    
    Returns:
        np.ndarray: predicted new state vector (shape: [n, 1]).
    """
    x = np.matrix(x).reshape(-1,1)
    x_new = (A @ x) + (B * u)
    return x_new


def covariance(A, P, Q):
    """
    - calculate the updated covariance matrix based on the state transition matrix and process noise covariance.
    
    Parameters:
        A (np.ndarray): state transition matrix (shape: [n, n]).
        P (np.ndarray): current covariance matrix of the state estimate (shape: [n, n]).
        Q (np.ndarray): process noise covariance matrix (shape: [n, n]).
    
    Returns:
        np.ndarray: updated covariance matrix (shape: [n, n]).
    """
    covariance = (A @ P @ A.T) + Q
    return covariance 


def OCV(soc):
    """
    - estimates ocv for a given soc using a polynomial regression model

    Parameters:
        soc (float or np.ndarray): soc value for which to calculate the ocv
    
    Returns:
        float or np.ndarray: estimated ocv
    """
    SOCOCV, _, _ = sococvRegression()
    OCV_value = np.polyval(SOCOCV, soc)
    return OCV_value


def g(x,D,u):
    """
    - calculate the terminal voltage based on the state vector, resistance, and control input

    Parameters:
        x (np.ndarray): state vector
        D (float): constant R0 value
        u (float): control input, representing current

    Returns:
        float: estimated terminal voltage 
    """
    terminal_voltage = OCV(x[0,0]) - x[1,0] - x[2,0] - D*u
    
    return terminal_voltage


def Cmatrix(SOCOCV, soc):
    """
    - compute C matrix used in the Kalman Gain by linearizing the mapping function
    - calculates the derivative of the SOC-OCV polynomial curve to determine how the ocv changes
    with respect to the soc and returns the resulting Jacobian matrix

    Parameters:
        SOCOCV (np.poly1d): polynomial function representing the ocv as a function of soc
        soc (float): current soc value at which to compute the derivative

    Returns:
        np.ndarray: the C matrix, which is a row vector containing:
            - the derivative of the curve at the given soc value
            - constant values for resistance components (-1 and -1) to represent certain parameters
    """
    
    dSOCOCV = np.polyder(SOCOCV)
    dOCV = np.polyval(dSOCOCV, soc)
    C_k = np.array([dOCV, -1, -1])
    
    return C_k

def terminalVoltageToOCV(Vt, x, D, u):
    return Vt + x[1, 0] + x[2, 0] + D*u

def ekf():  
    """
    Extended Kalman Filter (EKF) for estimating the State of Charge (SOC) of a battery based on measured terminal voltage and current
    
    This function performs the following steps:
    1. Load battery parameters from an Excel file.
    2. Define functions to interpolate battery model parameters.
    3. Perform polynomial regression to obtain the SOC-OCV curve.
    4. Initialize state variables and parameters.
    5. Read measurement data from a CSV file.
    6. Iterate through measurements to update state estimates using EKF.
        6.1 get current measurement, state estimate and battery parameters based on current soc
        6.2 compute time constants
        6.3 define state transition matrix A and control input matrix B
        6.4 predict new state and covariance
        6.5 compute estimated terminal voltage
        6.6 compute measurement Jacobian matrix C
        6.7 compute measurement residual
        6.8 compute measurement prediction covariance S
        6.9 compute kalman gain
        6.10 update state estimate and covariance matrix
    
    Returns:
        list: Estimated SOC values over time.
    """
    
    # step 1
    parameters = pd.read_excel("/Users/maishimozato/Documents/uoftEce2ndYear/research/dataCollection/battery_model.xlsx")
    
    # step 2
    R0_func = interp1d(parameters['SOC'], parameters['R0'], kind='linear', fill_value='extrapolate')
    R1_func = interp1d(parameters['SOC'], parameters['R1'], kind='linear', fill_value='extrapolate')
    R2_func = interp1d(parameters['SOC'], parameters['R2'], kind='linear', fill_value='extrapolate')
    C1_func = interp1d(parameters['SOC'], parameters['C1'], kind='linear', fill_value='extrapolate')
    C2_func = interp1d(parameters['SOC'], parameters['C2'], kind='linear', fill_value='extrapolate')
    
    # step 3
    SOCOCV, SOC, OCV = sococvRegression()
    plotSOCOCV(SOC, OCV, SOCOCV)
    
    # step 4
    x_pred = np.array([1, 0, 0]) # initial SOC, v1 and v2
    
    deltaT = 1 # time step (sec)
    Qnom = 5.8 * 3600 # capacity in amp-seconds
       
    # initial covariance matrix, representing the uncertainty in initial state estimate
    # off-diagonal values set to 0 unless there is known correlation between the states
    P_pred   = np.array([[0.001, 0, 0],
                      [0, 0.01, 0],
                      [0, 0, 0.01]])
    
    # process noise covariance matrix - uncertainties in model dynamics
    Q_k   = np.array([[1e-5, 0, 0],
                      [0, 1e-5, 0], 
                      [0, 0, 1e-5]])
    
    # measurement noise covariance
    R_k   = 2.5e-5
    
    SOC_estimations = []

    # step 5
    df = pd.read_csv('/Users/maishimozato/Documents/uoftEce2ndYear/research/dataCollection/B0005_TTD.csv')
    Vt_actual = df['Voltage_measured']
    current_actual = df['Current_measured']
    
    current_actual = - current_actual
    
    step = 1000
    indices = np.arange(0, len(current_actual), step)
    current_sample = current_actual[indices]
    
    length = len(current_actual)
    
    # step 6
    for k in range(1, 2500):
        
        # 6.1
        u = current_actual[k]
        v_measured = Vt_actual[k]
        soc = x_pred[0]
        print(soc)
        
        R0 = R0_func(soc)
        R1 = R1_func(soc)
        R2 = R2_func(soc)
        C1 = C1_func(soc)
        C2 = C2_func(soc)
        
        # 6.2
        tau1 = R1*C1
        tau2 = R2*C2
        
        # 6.3
        A_k = np.matrix([[1,0,0], 
                 [0, np.exp(-deltaT/(tau1)),0], 
                 [0,0, np.exp(-deltaT/(tau2))]])
        
        B_k = np.matrix([[-deltaT/Qnom], 
             [R1*(1 - np.exp(-deltaT/tau1))], 
             [R2*(1 - np.exp(-deltaT/tau2))]])
        
        # 6.4
        x_pred = f_function(A_k, B_k, x_pred, u) 
        P_pred = covariance(A_k, P_pred, Q_k) 
        
        # 6.5
        terminal_voltage = g(x_pred, R0, u)
        
        # 6.6
        C_k = Cmatrix(SOCOCV, soc)
        C_k = np.matrix(C_k) 
        
        # 6.7
        residual = v_measured - terminal_voltage
        
        # 6.8
        ones = np.matrix(np.ones((3,3)))
        S_k = (C_k @ P_pred @ C_k.T) + (R_k * ones)
        
        # 6.9
        KalmanGain = P_pred @ S_k @ C_k.T
        
        # 6.10
        x_pred = x_pred + (KalmanGain * int(residual))
        # ensure residual is a 1D array for matrix multiplication
        x_pred = np.array([x_pred[0,0], x_pred[1,0], x_pred[2,0]])
        
        SOC_estimations.append(x_pred[0])
        
        I = np.eye(3,3)
        P_pred = (I - (KalmanGain @ C_k)) @ (P_pred)
        
    return SOC_estimations

def plot(estimations, intervals):
    estimations = np.array(estimations)
    intervals = np.array(intervals)
    
    plt.figure(figsize=(10,6))
    plt.plot(intervals, estimations, marker='o')

def plot_results(SOC_estimations):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(SOC_estimations, label='Estimated SOC')
    plt.title('State of Charge Estimation')
    plt.xlabel('Time')
    plt.ylabel('SOC')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

SOC_estimations = ekf()
plot_results(SOC_estimations)
