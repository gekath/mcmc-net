import numpy as np


def softmax(data):
    '''
    Helper function to calculate softmax given some input data.
    '''
    return np.exp(data) / np.sum(np.exp(data), axis=0)

