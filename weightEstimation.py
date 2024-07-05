import matplotlib.pyplot as plt

weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

time_step = 1.0  # day
scale_factor = 4.0/10

#this filter requires us to correctly guess the rate of change
def predict_using_gain_guess(estimated_weight, gain_rate, do_print=False):     
    # storage for the filtered results
    estimates, predictions = [estimated_weight], []

    # most filter literature uses 'z' for measurements
    for z in weights: 
        # predict new position
        predicted_weight = estimated_weight + gain_rate * time_step

        # update filter 
        estimated_weight = predicted_weight + scale_factor * (z - predicted_weight)

        # save and log
        estimates.append(estimated_weight)
        predictions.append(predicted_weight)
        if do_print:
            print(f"Estimated weight: {estimated_weight:.2f}, Predicted weight: {predicted_weight:.2f}")

    return estimates, predictions

#takes into account the actual rate of change (measurement - estimate)
def predict_using_rate(estimated_weight, gain_rate, do_print=False):     
    # storage for the filtered results
    estimates, predictions = [estimated_weight], []

    # most filter literature uses 'z' for measurements
    for z in weights: 
        # predict new position
        predicted_weight = estimated_weight + gain_rate * time_step

        # update filter 
        residual = z - predicted_weight
        gain_rate += gain_scale * (residual/time_step)
        estimated_weight = predicted_weight + weight_scale * residual

        # save and log
        estimates.append(estimated_weight)
        predictions.append(predicted_weight)
        if do_print:
            print(f"Estimated weight: {estimated_weight:.2f}, Predicted weight: {predicted_weight:.2f}")

    return estimates, predictions

#predict the next measurement and rate of change based on the current estimate and how much we think it will change
#the new estimate = part way between the prediction and next measurement scaled by how accurate each is

initial_estimate = 160.0

# estimates, predictions = predict_using_gain_guess(
#     estimated_weight=initial_estimate, 
#     gain_rate=1, 
#     do_print=True
# )  

# # Plotting results
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(weights)+1), estimates, label='Estimated Weight')
# plt.plot(range(1, len(weights)+1), predictions, label='Predicted Weight', linestyle='--')
# plt.scatter(range(1, len(weights)+1), weights, color='red', marker='o', label='Measured Weight')
# plt.xlabel('Measurement Index')
# plt.ylabel('Weight')
# plt.title('Kalman Filter Weight Estimation')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

gain_rate = -1.0
weight_scale = 4.0/10.0
gain_scale = 1.0/3.0

estimates, predictions = predict_using_rate(
    estimated_weight=initial_estimate, 
    gain_rate=1, 
    do_print=True
)  

plt.figure(figsize=(10, 6))
plt.plot(range(len(weights)+1), estimates, label='Estimated Weight')
plt.plot(range(1, len(weights)+1), predictions, label='Predicted Weight', linestyle='--')
plt.scatter(range(1, len(weights)+1), weights, color='red', marker='o', label='Measured Weight')
plt.xlabel('Measurement Index')
plt.ylabel('Weight')
plt.title('Kalman Filter Weight Estimation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()