__author__ = 'bohaohan'
from Layer import *
from utilities import *


class FC(Layer):
    def __init__(self, input_, units):
        Layer.__init__(self)

        self.w = np.random.randn(len(input_), units)
        self.b = np.random.randn(1, units)
        self.output = 0
        self.b_input = input_

    def forward(self, b_input):
        self.b_input = b_input
        z1 = self.b_input.dot(self.w) + self.b
        # self.output = sigmoid(z1)
        return z1

    def backward(self, loss, lr=0.001, reg_lambda=0.0001):
        dw = self.b_input.T.dot(loss)
        dw += reg_lambda * self.w  # add penalty

        db = np.sum(loss, axis=0, keepdims=True)

        # top_loss = loss.dot(self.w.T) * sigmoid_driva(self.b_input)
        top_loss = loss.dot(self.w.T)

        self.w -= lr * dw
        self.b -= lr * db

        return top_loss