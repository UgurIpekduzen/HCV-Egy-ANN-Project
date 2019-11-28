import numpy as np
import math

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return 2 * sigmoid(2 * x) - 1

def relu(x):
    if x >= 0:
        return x
    return 0

def leakyRelu(x):
    if x >= 0:
        return x
    return x * 0.01

def swish(x):
    return x * sigmoid(x)

def softmax(x):
    return np.exp(x)/sum(np.exp(x))
