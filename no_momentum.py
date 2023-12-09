import numpy as np



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    # Real derivative of sigmoid function is sigmoid(a) * (1 - sigmoid(a)) but here i will pass argumnt x
    # as x = sigmoid(a)
    return x * (1 - x)


def initialize_weights(input_size, hidden_size, output_size):
    # TODO(11jolek11): How to manipulate initial values for testing purposes?
    np.random.seed(42)
    weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
    weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1
    return weights_input_hidden, weights_hidden_output

def train_neural_network(X, y, epochs, learning_rate, activasion, activasion_derivative):
    print(f"Input size {X.shape[1]}")
    input_size = X.shape[1]
    hidden_size = 2
    output_size = 1

    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden)
        hidden_layer_output = activasion(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        predicted_output = activasion(output_layer_input)

        # Backpropagation
        output_error = y - predicted_output
        output_delta = output_error * activasion_derivative(predicted_output)

        hidden_layer_error = output_delta.dot(weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * activasion_derivative(hidden_layer_output)

        # Update weights
        weights_hidden_output += learning_rate * hidden_layer_output.T.dot(output_delta)
        weights_input_hidden += learning_rate * X.T.dot(hidden_layer_delta)

    return weights_input_hidden, weights_hidden_output


def predict(X, weights_input_hidden, weights_hidden_output, activation):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = activation(output_layer_input)

    return predicted_output


if __name__ == "__main__":
    # FIXME(11jolek11): Network is not calculating correct predictions
    # XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    trained_weights_input_hidden, trained_weights_hidden_output = train_neural_network(X, y, epochs=100000, learning_rate=0.1, activasion=sigmoid, activasion_derivative=sigmoid_derivative)

    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = predict(test_input, trained_weights_input_hidden, trained_weights_hidden_output, sigmoid)

    print(predictions)

