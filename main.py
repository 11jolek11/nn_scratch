# Pytania do dr Turowskiej
#
# - czy liczba neuronow ukrytych tez ma byc parametrem symluacji?
# - Jak mamy przedstawic etapy uczenia sieci na wykresie?
# - Co ma sie znalezdz w sprawku


import numpy as np


class Net:
    def __init__(self, input_size, hidden_size, output_size, activation, activation_derivative) -> None:
        np.random.seed(42)
        self.weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
        self.weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.activation_derivative = activation_derivative

    def forward(self, X, y, learning_rate):
        # Forward pass
        hidden_layer_input = np.dot(X, self.weights_input_hidden)
        hidden_layer_output = self.activation(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        predicted_output = self.activation(output_layer_input)

        # Backpropagation
        output_error = y - predicted_output
        output_delta = output_error * self.activation_derivative(predicted_output)

        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.activation_derivative(hidden_layer_output)

        # Update weights
        self.weights_hidden_output += learning_rate * hidden_layer_output.T.dot(output_delta)
        self.weights_input_hidden += learning_rate * X.T.dot(hidden_layer_delta)

    def predict(self, x):
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = self.activation(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        predicted_output = self.activation(output_layer_input)

        return predicted_output


class Teacher:
    def __init__(self, model, learning_seq, ground_truth, max_epochs_number: int = 1000, lr: float = 0.001, size: tuple[int, int, int] = (2, 2, 1)):
        self.model = model
        self.learning_seq = learning_seq
        self.ground_truth = ground_truth
        self.max_epochs_number = max_epochs_number
        self.lr = lr
        self.size = size
        # self.stop_criteria = 


    def set_stop_criteria(self):
        pass

    def train(self):
        pass

