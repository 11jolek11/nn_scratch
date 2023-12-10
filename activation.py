import numpy as np


def binaryStep(x):
    ''' It returns '0' is the input is less then zero otherwise it returns one '''
    return np.heaviside(x,1)

def binaryStep_derivative(x):
    if x == 0:
        return 1
    else:
        return 0

def tanh(x):
    ''' It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1.'''

    return np.tanh(x)

def tanh_derivative(x):
    return 1 - tanh(x)**2

def relu(x):
    ''' It returns zero if the input is less than zero otherwise it returns the given input. '''
    return np.max(0, x)

def relu_derivative(x):
    return binaryStep(x)

# SOFTMAX

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Real derivative of sigmoid function is sigmoid(a) * (1 - sigmoid(a)) but here i will pass argumnt x
    # as x = sigmoid(a)
    return x * (1 - x)

