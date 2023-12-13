# Pytania do dr Turowskiej
#
# - czy liczba neuronow ukrytych tez ma byc parametrem symluacji?
# - Jak mamy przedstawic etapy uczenia sieci na wykresie?
# - Co ma sie znalezdz w sprawku


import numpy as np
from activation import relu_derivative, tanh
from no_momentum import sigmoid, sigmoid_derivative
import pandas as pd
import copy
import matplotlib.pyplot as plt
from activation import *




def mse(actual, predicted):
    return (actual - predicted) ** 2

def mse_grad(actual, predicted):
    return predicted - actual


class Net:
    def __init__(self, input_size, hidden_size, output_size, activation, activation_derivative, debug=True) -> None:
        if debug:
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
        # output_error = predicted_output - y
        output_delta = output_error * self.activation_derivative(predicted_output)

        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * self.activation_derivative(hidden_layer_output)

        # Update weights
        self.weights_hidden_output += learning_rate * hidden_layer_output.T.dot(output_delta)
        self.weights_input_hidden += learning_rate * X.T.dot(hidden_layer_delta)

    def predict(self, x, activation):
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = self.activation(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        predicted_output = activation(output_layer_input)

        return predicted_output


class NetMomentum:
    def __init__(self, input_size, hidden_size, output_size, activation, activation_derivative, debug=True) -> None:
        if debug:
            np.random.seed(42)
        self.weights_input_hidden = 1 * np.random.random((input_size, hidden_size))
        self.weights_hidden_output = 1 * np.random.random((hidden_size, output_size))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.activation_derivative = activation_derivative
        # Init momentum
        self.momentum_weights_input_hidden = np.zeros_like(self.weights_input_hidden)
        self.momentum_weights_hidden_output = np.zeros_like(self.weights_hidden_output)


    def forward(self, X, y, learning_rate, momentum):
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
        self.momentum_weights_hidden_output = momentum * self.momentum_weights_hidden_output + learning_rate * hidden_layer_output.T.dot(output_delta)
        self.momentum_weights_input_hidden = momentum * self.momentum_weights_input_hidden + learning_rate * X.T.dot(hidden_layer_delta)

        self.weights_hidden_output += self.momentum_weights_hidden_output
        self.weights_input_hidden += self.momentum_weights_input_hidden

    def predict(self, x, activation):
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = self.activation(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        predicted_output = activation(output_layer_input)

        return predicted_output


class Teacher:
    def __init__(self, model:Net | NetMomentum, learning_seq, validation_seq, ground_truth, error_function, momentum=0.9, max_epochs_number: int = 1000, lr: float = 0.1, size: tuple[int, int, int] = (2, 2, 1), stop_criteria=lambda _: (False, None)):
        self.model = model
        self.learning_seq = learning_seq
        self.validation_seq = validation_seq
        self.ground_truth = ground_truth
        self.max_epochs_number = max_epochs_number
        self.lr = lr
        self.size = size
        self.stop_criteria = stop_criteria
        self.momentum = momentum
        self.error_function = error_function

    def get_model(self):
        return self.model

    def train(self):
        train_history = pd.DataFrame(columns=["model", "epoch", "error"])
        model = None
        error_history = []
        # TODO(11jolek11): Implement stop criterium
        for epoch in range(self.max_epochs_number):
            if isinstance(self.model, NetMomentum):
                self.model.forward(self.learning_seq, self.ground_truth, self.lr, self.momentum)
            if isinstance(self.model, Net):
                self.model.forward(self.learning_seq, self.ground_truth, self.lr)

            model = copy.copy(self.model)

            error = mse(self.ground_truth, model.predict(self.learning_seq, sigmoid))

            error_history.append(np.mean(error))

            temp_dict = {"model": model, "epoch": epoch, "error": error}

            train_history._append(temp_dict, ignore_index=True)

            temp = self.stop_criteria(train_history)
            if temp[0]:
                 self.model = temp[1]
                 break
        plt.plot(error_history)
        plt.show()

        print(f"{len(error_history)}")

if __name__ == "__main__":
    # test_model = Net(2, 2, 1, sigmoid, sigmoid_derivative)
    test_model = NetMomentum(2, 2, 1, sigmoid, sigmoid_derivative)
    # XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    test_input = np.array([1, 1])

    test_teacher = Teacher(test_model, X, [], y, None, max_epochs_number=10000)

    test_teacher.train()

    trained_model = test_teacher.get_model()

    predictions = trained_model.predict(test_input, sigmoid)

    print(predictions)

