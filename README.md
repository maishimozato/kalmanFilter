# kalmanFilter
Kalman Filters: A Brief Overview
Kalman Filters are powerful tools for state estimation, particularly useful when dealing with noisy measurements and uncertain dynamics. They provide an optimal way to estimate the state of a system over time based on a series of measurements.

Key Concepts
State Estimation
Kalman Filters estimate the state of a system based on noisy data. The estimation is refined by combining predictions from a system model with actual measurements.

System Components
System: What we want to estimate.
State: The actual (hidden) value of the system.
Measurement: Observed value of the system, often inaccurate.
Residual: Difference between the measured and predicted values.
Algorithm Steps
Initialization
Initialize the state of the filter.
Establish an initial belief in the state.
Predict
Use the system model to predict the state at the next time step.
Adjust the belief to account for prediction uncertainties.
Update
Obtain a measurement and assess its accuracy.
Compute the residual between the estimated state and measurement.
Adjust the state estimate based on the measurement using the Kalman gain.
Kalman Filter Algorithm
Predict Step
State Propagation
𝑥
^
𝑘
−
=
𝐴
𝑥
^
𝑘
−
1
+
𝐵
𝑢
𝑘
x
^
  
k
−
​
 =A 
x
^
  
k−1
​
 +Bu 
k
​
 

A: Matrix describing how the system evolves independently.
B: Matrix showing how control inputs affect the system.
𝑢
𝑘
u 
k
​
 : Control inputs.
State Covariance Update
𝑃
𝑘
−
=
𝐴
𝑃
𝑘
−
1
𝐴
𝑇
+
𝑄
𝑘
P 
k
−
​
 =AP 
k−1
​
 A 
T
 +Q 
k
​
 

𝑃
𝑘
−
P 
k
−
​
 : Predicted state covariance matrix.
Q_k: Process noise covariance matrix, accounting for unpredicted changes.
Update Step
Measurement Incorporation
𝐾
𝑘
=
𝑃
𝑘
−
𝐻
𝑘
𝑇
(
𝐻
𝑘
𝑃
𝑘
−
𝐻
𝑘
𝑇
+
𝑅
𝑘
)
−
1
K 
k
​
 =P 
k
−
​
 H 
k
T
​
 (H 
k
​
 P 
k
−
​
 H 
k
T
​
 +R 
k
​
 ) 
−1
 

𝐾
𝑘
K 
k
​
 : Kalman Gain.
H_k: Observation matrix, transforming state vector 
𝑥
^
𝑘
−
x
^
  
k
−
​
  to predicted measurement.
R_k: Measurement noise covariance matrix.
State Update
𝑥
^
𝑘
=
𝑥
^
𝑘
−
+
𝐾
𝑘
(
𝑧
𝑘
−
𝐻
𝑘
𝑥
^
𝑘
−
)
x
^
  
k
​
 = 
x
^
  
k
−
​
 +K 
k
​
 (z 
k
​
 −H 
k
​
  
x
^
  
k
−
​
 )

𝑥
^
𝑘
x
^
  
k
​
 : Updated state estimate.
𝑧
𝑘
z 
k
​
 : Actual measurement.
Covariance Update
𝑃
𝑘
=
(
𝐼
−
𝐾
𝑘
𝐻
𝑘
)
𝑃
𝑘
−
P 
k
​
 =(I−K 
k
​
 H 
k
​
 )P 
k
−
​
 

𝑃
𝑘
P 
k
​
 : Updated state covariance matrix.
Observation Model
The observation model predicts what measurements we should observe based on the predicted state 
𝑥
^
𝑘
−
x
^
  
k
−
​
 .

𝑧
𝑘
=
𝐻
𝑘
𝑥
^
𝑘
−
+
noise
z 
k
​
 =H 
k
​
  
x
^
  
k
−
​
 +noise

Noise: Represents uncertainty in the measurement.
Conclusion
Kalman Filters are versatile tools for state estimation, providing accurate estimates even in the presence of noise and uncertainty. By integrating system dynamics and measurement data, they offer robust solutions across various applications.
