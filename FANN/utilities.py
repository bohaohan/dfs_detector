__author__ = 'bohaohan'
import numpy as np


def softmax(output_array):
    logits_exp = np.exp(output_array)
    return logits_exp / np.sum(logits_exp, axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))