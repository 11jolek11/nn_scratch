# Pytania do dr Turowskiej
#
# - czy liczba neuronow ukrytych tez ma byc parametrem symluacji?
# - Jak mamy przedstawic etapy uczenia sieci na wykresie?
# - Co ma sie znalezdz w sprawku


import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Real derivative of sigmoid function is sigmoid(a) * (1 - sigmoid(a)) but here i will pass argumnt x
    # as x = sigmoid(a)
    return x * (1 - x)


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


class NetMomentum:
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

    def predict(self, x):
        hidden_layer_input = np.dot(x, self.weights_input_hidden)
        hidden_layer_output = self.activation(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        predicted_output = self.activation(output_layer_input)

        return predicted_output


class Teacher:
    def __init__(self, model:Net | NetMomentum, learning_seq, validation_seq, ground_truth, momentum=0.9, max_epochs_number: int = 1000, lr: float = 0.1, size: tuple[int, int, int] = (2, 2, 1), stop_criteria=False):
        self.model = model
        self.learning_seq = learning_seq
        self.validation_seq = validation_seq
        self.ground_truth = ground_truth
        self.max_epochs_number = max_epochs_number
        self.lr = lr
        self.size = size
        self.stop_criteria = stop_criteria
        self.momentum = momentum

    def get_model(self):
        return self.model

    def train(self):
        # TODO(11jolek11): Implement stop criterium
        for epoch in range(self.max_epochs_number):
            self.model.forward(self.learning_seq, self.ground_truth, self.lr, self.momentum)


if __name__ == "__main__":
    # test_model = Net(2, 2, 1, sigmoid, sigmoid_derivative)
    test_model = NetMomentum(2, 2, 1, sigmoid, sigmoid_derivative)
    # XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    test_teacher = Teacher(test_model, X, [], y, max_epochs_number=100000)

    test_teacher.train()

    trained_model = test_teacher.get_model()

    predictions = trained_model.predict(X)

    print(predictions)

