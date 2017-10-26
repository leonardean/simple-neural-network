import sys
import json
from pprint import pprint
import DataProcessor
import NeuralNetwork
import numpy as np
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
train_features, train_targets = data_processor.get_train_data()
val_features, val_targets = data_processor.get_val_data()

# Initialize NeuralNetwork
N_i = train_features.shape[1]
network = NeuralNetwork.NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)


losses = {'train': [], 'validation': []}

def MSE(y, Y):
    return np.mean((y-Y)**2)

for ii in range(iterations):
    # pick 128 random records from training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']

    network.train(X, y)

    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)

    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

# Store weights
weights_input_to_hidden, weights_hidden_to_output = network.get_weights()
np.save('weights_input_to_hidden', weights_input_to_hidden)
np.save('weights_hidden_to_output', weights_hidden_to_output)

# Plot losses
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
_ = plt.ylim()
plt.show()
