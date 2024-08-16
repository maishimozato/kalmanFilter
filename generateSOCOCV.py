import pandas as pd
import numpy as np
from scipy import interpolate

# soc and ocv files taken from data I collected with an ocv discharge test

# from the battery being discharged by 1.16A C/5 current
soc_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]  
ocv_values = [4.21, 4.033998, 4.027619, 3.97486, 3.870887, 3.84041, 3.825591, 3.738765, 3.077883, 2.99763, 2.78564]

# from the battery being discharged by 1.45A C/4 current
soc_values2 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]  
ocv_values2 = [4.16, 4.035215, 4.018772, 3.934327, 3.874745, 3.82052, 3.676826, 3.618304, 3.160079, 2.812154, 2.7473]

# Define the new SOC range with more values (linear interpolation)
soc_new = np.linspace(0.0, 1.0, num=100)  
# Extending SOC slightly beyond the original range
soc_new2 = np.linspace(0.0, 1.0, num=100)

# The OCV values are extrapolated based on the trend of the original OCV data.
# This means that the extrapolation considers the existing relationship between SOC and OCV values and extends that relationship beyond the original SOC range.
ocv_interp = interpolate.interp1d(soc_values, ocv_values, fill_value='extrapolate')
ocv_new = ocv_interp(soc_new)

ocv_interp2 = interpolate.interp1d(soc_values2, ocv_values2, fill_value='extrapolate')
ocv_new2 = ocv_interp2(soc_new2)

# Create a DataFrame to hold the new SOC and OCV values
df = pd.DataFrame({
    'SOC': soc_new,
    'OCV': ocv_new
})

df2 = pd.DataFrame({
    'SOC': soc_new2,
    'OCV': ocv_new2
})

# Save the DataFrame to an Excel file
df.to_excel('/Users/maishimozato/Documents/uoft ece - 2nd year/research/dataCollection/SOCOCV.xlsx', index=False)
df2.to_excel('/Users/maishimozato/Documents/uoft ece - 2nd year/research/dataCollection/SOCOCV2.xlsx', index=False)

print("Extrapolated values saved")
