import numpy as np
from no_momentum import sigmoid, sigmoid_derivative, initialize_weights


def train_neural_network_with_momentum(X, y, epochs, learning_rate, momentum):
    input_size = X.shape[1]
    hidden_size = 2
    output_size = 1

    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    # Init momentum
    momentum_weights_input_hidden = np.zeros_like(weights_input_hidden)
    momentum_weights_hidden_output = np.zeros_like(weights_hidden_output)

    for epoch in range(epochs):
        # Forward pass
        hidden_layer_input = np.dot(X, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        predicted_output = sigmoid(output_layer_input)

        # Backpropagation
        output_error = y - predicted_output
        output_delta = output_error * sigmoid_derivative(predicted_output)

        hidden_layer_error = output_delta.dot(weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

        # Update weights with momentum
        momentum_weights_hidden_output = momentum * momentum_weights_hidden_output + learning_rate * hidden_layer_output.T.dot(output_delta)
        weights_hidden_output += momentum_weights_hidden_output

        momentum_weights_input_hidden = momentum * momentum_weights_input_hidden + learning_rate * X.T.dot(hidden_layer_delta)
        weights_input_hidden += momentum_weights_input_hidden

    return weights_input_hidden, weights_hidden_output


def predict_with_momentum(X, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    return predicted_output


if __name__ == "__main__":
    # XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    trained_weights_input_hidden, trained_weights_hidden_output = train_neural_network_with_momentum(X, y, epochs=10000, learning_rate=0.1, momentum=0.9)

    test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = predict_with_momentum(test_input, trained_weights_input_hidden, trained_weights_hidden_output)

    print(predictions)
