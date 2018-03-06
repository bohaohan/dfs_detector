__author__ = 'bohaohan'
from utilities import *
from Layer import *


class Sigmoid(Layer):

    def __init__(self):
        Layer.__init__(self)

    def forward(self, b_input):
        return sigmoid(b_input)

    def backward(self, loss, lr):
        return loss * sigmoid_driva(loss)