__author__ = 'bohaohan'
import numpy as np


def softmax(output_array):
    logits_exp = np.exp(output_array)
    return logits_exp / np.sum(logits_exp, axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_driva(x):
    # drivarive of sigmoid function
    return np.exp(-x) / ((1+np.exp(-x)) ** 2)


def categorical_crossentropy(output, target):

    assert output.shape[1] == target.shape[1], "model output" + str(output.shape) + \
                                               " should have same shape with target" + str(target.shape)

    # output /= np.sum(output, axis=len(output.shape) - 1, keepdims=True)
    output = softmax(output)
    # manual computation of crossentropy
    epsilon = np.finfo(float).eps
    output = np.clip(output, epsilon, 1. - epsilon)

    return np.mean(- np.sum(target * np.log(output), axis=len(output.shape) - 1)), target - output

