__author__ = 'bohaohan'
from utilities import *


class Fann_Model:
    def __init__(self, input_shape):
        self.layers = []
        self.outputs = []
        self.gradient = []
        self.lr = 0.001
        self.input = None

    def add(self, layer):
        self.layers.append(layer)
        self.outputs.append([])
        self.gradient.append([])

    def forward_all(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                self.outputs[i] = layer.forward(self.input)
                continue
            self.outputs[i] = layer.forward(self.outputs[i - 1])

    def back_propagation(self):
        for i, layer in reversed(list(enumerate(self.layers))):
            self.gradient[i] = layer.backward(self.gradient[i + 1], self.lr)

    def get_accuracy(self, output, target):
        # print output
        # print target
        _ = np.array([i.argmax() for i in output]).T
        y = np.array([i.argmax() for i in target]).T

        total = len(output)
        right = (_ == y).sum()

        return float(right) / float(total)

    def fit(self, x, y, max_epoch=5, lr=0.0001):
        self.input = x
        self.lr = lr
        if len(self.gradient) == len(self.outputs):
            self.gradient.append([])
            print len(self.outputs), len(self.gradient)

        assert len(x.shape) == 2, "X should be in 2 dimensions like (40, 2)"
        assert len(y.shape) == 2, "y should be in 2 dimensions like (40, 1)"

        losses = []
        for epoch in range(max_epoch):
            # forward
            self.forward_all()
            loss, gradient = categorical_crossentropy(self.outputs[-1], y)

            accuracy = self.get_accuracy(self.outputs[-1], y)
            print "Current epoch:", epoch, "Loss:", loss, "accuracy:", accuracy
            losses.append(loss)

            self.gradient[-1] = gradient

            # backward
            self.back_propagation()

        return losses

    def predict(self, x):
        assert len(x.shape) == 2, "X should be in 2 dimensions like (40, 2)"
        self.input = x
        self.forward_all()
        return self.outputs[-1]

    def evaluate(self, x, y):
        pred = self.predict(x)
        accuracy = self.get_accuracy(pred, y)
        return accuracy