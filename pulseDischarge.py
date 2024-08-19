import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/Users/maishimozato/Documents/uoftEce2ndYear/research/dataCollection/pulseDischarge.csv')  

sampling_rate = 0.2 #seconds
df['Time (hours)'] = np.arange(len(df)) * sampling_rate / 3600  


plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(df['Time (hours)'], df['Voltage'], label='Voltage (V)', color='blue')
plt.xlabel('Time (hours)')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs. Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df['Time (hours)'], df['Current'], label='Current (A)', color='red')
plt.xlabel('Time (hours)')
plt.ylabel('Current (A)')
plt.title('Current vs. Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
