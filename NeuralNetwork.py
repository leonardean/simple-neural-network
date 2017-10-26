import numpy as np

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate,
            weights_input_to_hidden=None, weights_hidden_to_output=None):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        if type(weights_input_to_hidden).__name__ == 'NoneType' and type(weights_hidden_to_output).__name__ == 'NoneType':
            self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                           (self.input_nodes, self.hidden_nodes))
            self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                           (self.hidden_nodes, self.output_nodes))
        else:
            self.weights_input_to_hidden = weights_input_to_hidden
            self.weights_hidden_to_output = weights_hidden_to_output

        self.lr = learning_rate

        def sigmoid(x):
            return 1 / (1 + np.exp( -x ))

        def sigmoid_prime(x):
            return sigmoid(x) * (1 - sigmoid(x))

        def linear(x):
            return x

        def linear_prime(x):
            return x ** 0
        # Activation functions
        self.activation_function = sigmoid
        self.activation_function_prime = sigmoid_prime
        self.activation_function2 = linear
        self.activation_function_prime2 = linear_prime

    def train(self, features, targets):
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            # Forward Pass
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)

            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
            final_outputs = self.activation_function2(final_inputs)

            # Backward Pass
            error = y - final_outputs
            output_error_term = error * self.activation_function_prime2(final_outputs)

            hidden_error = np.dot(output_error_term, self.weights_hidden_to_output.T)
            hidden_error_term = hidden_error * self.activation_function_prime(hidden_inputs)

            # Weight steps
            delta_weights_i_h += hidden_error_term * X[:, None]
            delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = self.activation_function2(final_inputs)

        return final_outputs

    def get_weights(self):
        return self.weights_input_to_hidden, self.weights_hidden_to_output
