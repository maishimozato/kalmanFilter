import matplotlib.pyplot as plt
import numpy as np

from numpy.random import randn

def gen_data(x0, dx, count, noise_factor):
    return [x0 + dx*i + randn()*noise_factor for i in range(count)]

def g_h_filter(data, x0, dx, g, h, dt):
    
    """
    Performs g-h filter on 1 state variable with a fixed g and h.

    'data' contains the data to be filtered.
    'x0' is the initial value for our state variable
    'dx' is the initial change rate for our state variable
    'g' is the g-h's g scale factor
    'h' is the g-h's h scale factor
    'dt' is the length of the time step 
    """ 
    
    x_est = x0
    results = []
    for z in data:
        #prediction step
        x_pred = x_est + (dx*dt)

        # update step
        residual = z - x_pred
        dx = dx + h * (residual) / dt
        x_est = x_pred + g * residual
        results.append(x_est)
    return np.array(results)

weights = gen_data(5, 2, 100, 100)


# Perform g-h filter
filtered_data = g_h_filter(data=weights, x0=5., dx=2., g=0.2, h=0.02, dt=1.)

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(range(len(weights)), weights, marker='o', linestyle='-', color='b', label='Measured Weight')
plt.plot(range(len(weights)), filtered_data, marker='x', linestyle='--', color='r', label='Filtered Weight (Estimate)')
plt.xlabel('Measurement Index')
plt.ylabel('Weight')
plt.title('G-H Filter Weight Estimation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("measurements:", weights)
print("estimates:", filtered_data)
        
    