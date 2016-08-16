import numpy as np
import scipy.io as sios


def softmax(data):
    '''
    Helper function to calculate softmax given some input data.
    '''
    return np.exp(data) / np.sum(np.exp(data), axis=0)


def load_mnist(directory, class_num, set_type='train'):
    # class_string = 'train%d' %(class_num) + '.txt'
    class_string = directory + '/' + set_type + '%d.txt' % (class_num)
    data = np.loadtxt(class_string) / 255

    n_cases, n_dim = data.shape

    return data, n_cases


def convert_to_text(filename):
    data = sio.loadmat(filename)

    for i in range(self.num_classes):
        class_string = 'test%d' % (i)
        print(class_string)
        class_data = data[class_string]
        np.savetxt(class_string + '.txt', class_data)