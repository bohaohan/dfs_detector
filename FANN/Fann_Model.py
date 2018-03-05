__author__ = 'bohaohan'


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
            self.gradient[i] = layer.backward(self.outputs[i + 1], self.lr)

    def fit(self):
        pass
