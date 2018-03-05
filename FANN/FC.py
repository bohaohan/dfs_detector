__author__ = 'bohaohan'
from Layer import *
from utilities import *


class FC(Layer):
    def __init__(self, input, units):
        Layer.__init__(self)

        self.w = np.random.randn(len(input), units)
        self.b = np.random.randn(1, units)

    def forward(self, b_input):
        self.b_input = b_input
        z1 = self.b_input.dot(self.w) + self.b
        return sigmoid(z1)

    def backward(self, loss, lr=0.001, reg_lambda=0.1):
        dw = self.b_input.T.dot(loss)
        db = np.sum(loss, axis=0, keepdims=True)

        top = loss.dot(self.w.T) * (1 - np.power(self.b_input, 2))
        dw += reg_lambda * self.w

        self.w -= lr * dw
        self.b -= lr * db

        return top