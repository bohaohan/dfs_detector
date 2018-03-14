__author__ = 'bohaohan'
from Layer import *
from utilities import *


class FC(Layer):
    def __init__(self, units):
        Layer.__init__(self)

        self.w = None
        self.b = 2 * np.random.random_sample((1, units)) - 1 # Vector of the weights given to the bias term by each neuron
        self.output = 0
        self.b_input = None
        self.units = units

    def forward(self, b_input):
        self.b_input = b_input
        if self.w is None:
            # 2D matrix of weights initialized with random numbers between -1 and +1. The size of the matrix is given by 'Input Size' x 'Units in the Layer'
            self.w = 2 * np.random.random_sample((self.b_input.shape[1], self.units)) - 1
        z1 = self.b_input.dot(self.w) + self.b
        # self.output = sigmoid(z1)
        return z1

    def backward(self, loss, lr=0.001, reg_lambda=0.0001):
        dw = self.b_input.T.dot(loss)
        dw += reg_lambda * self.w  # add penalty

        db = np.sum(loss, axis=0, keepdims=True)

        top_loss = loss.dot(self.w.T)

        self.w += lr * dw
        self.b += lr * db

        return top_loss