import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as s 
import seaborn as sns

from scipy.interpolate import interp1d

def sococvRegression():
    sococv = pd.read_excel("/Users/maishimozato/Documents/uoftEce2ndYear/research/dataCollection/SOCOCV.xlsx")
    SOC = sococv['SOC'].values
    OCV = sococv['OCV'].values
    
    #fit a polynomial of degree 11 to the data
    coeffs = np.polyfit(SOC, OCV, 11)
    #create a polynomial representing ocv as a function of soc
    SOCOCV = np.poly1d(coeffs)
    
    return coeffs, SOCOCV, SOC, OCV

def plotSOCOCV(SOC, OCV, SOCOCV):
    #generate a set of evenly spaced values between the min and max values of soc data
    SOC_fit = np.linspace(SOC.min(), SOC.max(), 500)
    #compute corresponding ocv values for soc using polynomial function
    OCV_fit = SOCOCV(SOC_fit)
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,6))
    
    plt.scatter(SOC, OCV, label="Original Data", s=15, color='blue', alpha=0.7)
    plt.plot(SOC_fit, OCV_fit, color='red', linewidth=2.5, linestyle='--', label='Polynomial Fit')
    
    # Enhance plot with additional formatting
    plt.xlabel('State of Charge (SOC)', fontsize=14)
    plt.ylabel('Open Circuit Voltage (OCV)', fontsize=14)
    plt.title('SOC vs OCV with Polynomial Fit from C/5 OCV discharge test', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    return

#takes in the soc ocv curve and current state soc value
#returns the derivative of the curve for matrix C
def Cmatrix(coeffs, soc):
    #coefficients of the derivative - rate of change of ocv with respect to soc
    dSOCOCV = np.polyder(coeffs)
    
    #evaluate the polynomial at x = soc (slope of ocv soc curve at that point)
    dOCV = np.polyval(dSOCOCV, soc)
    C_x = np.array([[dOCV, -1, -1]])
    return C_x

#the function to predict the new state value based on current state and control input (current) 
def f_function(A, B, x, u):
    x_new = (A @ np.matrix(x).reshape(3, 1)) + (B * u)
    return x_new

def covariance(A, P, Q):
    covariance = (A @ P @ A.T) + Q
    return covariance 

# Function that relates soc to ocv - non linearity
def OCV(soc):
    coeffs, SOCOCV, SOC, OCV = sococvRegression()
    OCV_value = SOCOCV(soc)
    return OCV_value

# Function that relates ocv to the terminal voltage
def g(x,D,u):
    terminal_voltage = OCV(x[0,0]) - x[1, 0] - x[2, 0] - D*u
    #x[0,0] returns the scalar value while x[0] returns the 1D array of that value
    return terminal_voltage

def terminalVoltageToOCV(Vt, x, D, u):
    return Vt + x[1, 0] + x[2, 0] + D*u

def ekf():  
    parameters = pd.read_excel("/Users/maishimozato/Documents/uoftEce2ndYear/research/dataCollection/battery_model.xlsx")
    
    #initial SOC
    SOC_init = 1 
    #state space x - parameter initialization
    x_pred = np.array([[SOC_init], [0], [0]]) #shape is (3,1)
    
    # these will need to be calculated using the pulse discharge test
    #functions that interpolate between the given data
    R0_func = interp1d(parameters['SOC'], parameters['R0'], kind='linear', fill_value='extrapolate')
    R1_func = interp1d(parameters['SOC'], parameters['R1'], kind='linear', fill_value='extrapolate')
    R2_func = interp1d(parameters['SOC'], parameters['R2'], kind='linear', fill_value='extrapolate')
    C1_func = interp1d(parameters['SOC'], parameters['C1'], kind='linear', fill_value='extrapolate')
    C2_func = interp1d(parameters['SOC'], parameters['C2'], kind='linear', fill_value='extrapolate')
    
    coeffs, SOCOCV, SOC, OCV = sococvRegression()
    plotSOCOCV(SOC, OCV, SOCOCV)

    deltaT = 1
    #capacity in amp-seconds to match time intervals
    Qnom = 5.8 * 3600 
    
    R_k = 2.5e-5
    P_pred = np.array([[0.025, 0, 0],
                   [0, 0.01, 0],
                   [0, 0, 0.01]])
    Q_x = np.array([[1.0, 1e-6, 0],
                   [0, 1e-5, 0],
                   [0, 0, 1e-5]])
    
    SOC_estimations = []
    Vterminal_estimations = []
    Vterminal_error = []

    df = pd.read_csv('/Users/maishimozato/Documents/uoftEce2ndYear/research/dataCollection/B0005_TTD.csv')
    Vt_actual = df['Voltage_measured']
    current_actual = df['Current_measured']
    
    step = 1000
    indices = np.arange(0, len(current_actual), step)
    current_sample = current_actual[indices]
    
    length = len(current_sample)
    
    for k in range(length):
        u = current_actual[k]
        soc = x_pred[0, 0]
        
        #scalar values
        R0 = R0_func(soc)
        R1 = R1_func(soc)
        R2 = R2_func(soc)
        C1 = C1_func(soc)
        C2 = C2_func(soc)
        
        #time intervals in seconds
        tau1 = R1*C1
        tau2 = R2*C2
        #state transition matrix
        A = np.matrix([[1,0,0], 
                 [0, np.exp(-deltaT/(tau1)),0], 
                 [0,0, np.exp(-deltaT/(tau2))]])
        #shape is (3,3)
        
        # identity matrix - no change in the state model from the previous state
        B = np.matrix([[-deltaT/Qnom], 
             [R1*(1 - np.exp(-deltaT/tau1))], 
             [R2*(1 - np.exp(-deltaT/tau2))]])
        #shape is (3,1)
        
        x_pred = f_function(A, B, x_pred, u) #shape is (3,1)
        #covariance/uncertainty associated with state estimate
        P_pred = covariance(A, P_pred, Q_x) #shape is (3,3)
        
        # estimated terminal voltage based on estimated soc
        terminal_voltage = g(x_pred, R0, u) #scalar
        Vterminal_estimations.append(terminal_voltage)
        
        C_x = Cmatrix(coeffs, soc) #shape is (1,3)
        
        residual = Vt_actual[k] - terminal_voltage #scalar
        Vterminal_error.append(residual)
        
        S_x = (C_x @ P_pred @ C_x.T) + (R_k * np.eye(3)) #shape is (3,3)
        
        KalmanGain = P_pred @ np.linalg.inv(S_x) @ C_x.T #shape is (3,1)
        
        #update the state estimation and covariance matrix
        # Ensure residual is a 1D array for matrix multiplication
        x_pred = x_pred + (KalmanGain * residual)
        SOC_estimations.append(x_pred[0,0])
        
        I = np.eye(3)
        P_pred = (I - (KalmanGain @ C_x)) @ (P_pred)
        
    return SOC_estimations, Vterminal_estimations, Vterminal_error

def plot(estimations, intervals):
    estimations = np.array(estimations)
    intervals = np.array(intervals)
    
    if estimations.ndim != 1:
        raise ValueError("estimations must be a 1D array")
    if intervals.ndim != 1:
        raise ValueError("intervals must be a 1D array")
    
    plt.figure(figsize=(10,6))
    plt.plot(intervals, estimations, marker='o')

def plot_results(SOC_estimations, Vterminal_estimations, Vterminal_error):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(3, 1, 1)
    plt.plot(SOC_estimations, label='Estimated SOC')
    plt.title('State of Charge Estimation')
    plt.xlabel('Time')
    plt.ylabel('SOC')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(Vterminal_estimations, label='Estimated Terminal Voltage')
    plt.title('Terminal Voltage Estimation')
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(Vterminal_error, label='Voltage Error')
    plt.title('Voltage Error')
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

SOC_estimations, Vterminal_estimations, Vterminal_error = ekf()
plot_results(SOC_estimations, Vterminal_estimations, Vterminal_error)

# df = pd.read_csv('/Users/maishimozato/Documents/uoft ece - 2nd year/research/B0005_TTD.csv')
# RecordingTime            = df['Time']
# Measured_Voltage         = df['Voltage_measured']
# Measured_Current         = df['Current_measured']
# Measured_Temperature     = df['Temperature_measured'] 

# plt.figure(figsize=(12, 6))

# # Plot SOC estimations vs. actual SOC measurements
# plt.subplot(2, 1, 1)
# plt.plot(socEstimated, label='Estimated SOC', linestyle='--')
# plt.xlabel('Time Step')
# plt.ylabel('State of Charge (SOC)')
# plt.legend()
# plt.title('SOC Estimations vs. Actual Measurements')

# # Plot Terminal Voltage estimations vs. actual terminal voltage measurements
# plt.subplot(2, 1, 2)
# plt.plot(df['Voltage_measured'], label='Actual Voltage')
# plt.plot(VterminalEstimated, label='Estimated Voltage', linestyle='--')
# plt.xlabel('Time Step')
# plt.ylabel('Terminal Voltage (V)')
# plt.legend()
# plt.title('Terminal Voltage Estimations vs. Actual Measurements')

# plt.tight_layout()
# plt.show()
