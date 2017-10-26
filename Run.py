import json
from pprint import pprint
import DataProcessor
import NeuralNetwork
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get training parameters
with open('networkConfig.json') as config_file:
    config = json.load(config_file)
pprint(config)

iterations = config['iterations']
learning_rate = config['learning_rate']
hidden_nodes = config['hidden_nodes']
output_nodes = config['output_nodes']

# Get data
data_processor = DataProcessor.DataProcessor('Bike-Sharing-Dataset/hour.csv')
data_processor.virtualize()
data_processor.normalize()
data_processor.split()
test_features, test_targets, test_data = data_processor.get_test_data()
scaled_features = data_processor.get_scaled_features()
orig_data = data_processor.get_orig_data()

mean, std = scaled_features['cnt']

# Initialize network
weights_input_to_hidden = np.load('weights_input_to_hidden.npy')
weights_hidden_to_output = np.load('weights_hidden_to_output.npy')
N_i = test_features.shape[1]
network = NeuralNetwork.NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate,
            weights_input_to_hidden=weights_input_to_hidden,
            weights_hidden_to_output=weights_hidden_to_output)

# Run network prediction
predictions = network.run(test_features).T * std + mean

# Plot prediction and ground trueth
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(orig_data.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.show()
