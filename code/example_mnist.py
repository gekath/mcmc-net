from rbm4 import *
from utils import *
import numpy as np
import sys
import scipy.io as sio


class TestMNIST():

    def __init__(self, directory, num_classes):
        self.directory = directory
        self.num_classes = num_classes
        self.rmse = None
        self.params = None
        self.all_weights = None
        self.all_vis_bias = None
        self.all_hid_bias = None
        self.trainer = None

    def train_all(self, num_epochs, batchsize, sampler_type='Gibbs'):

        self.all_vis_bias = []
        self.all_hid_bias = []
        self.all_weights = []

        for class_num in range(self.num_classes):

            print('Beginning to load MNIST dataset ' + str(class_num))
            data, n_cases = load_mnist(self.directory, class_num)
            print('Finished loading MNIST dataset ' + str(class_num))

            print('Initializing RBM' + str(class_num))
            model = RBM(784, 1000)
            # rbm.params[:] = np.random.uniform(-1./10, 1./10, len(rbm.params))

            print('Training RBM' + str(class_num))
            self.trainer = Trainer(model)

            rmse, params = self.trainer.train(data, num_epochs,
                                         batchsize=batchsize,
                                         sampler=sampler_type)
            weights, vis_bias, hid_bias = params

            self.all_weights.append(weights)
            self.all_vis_bias.append(vis_bias)
            self.all_hid_bias.append(hid_bias)

        # return np.array(self.all_weights), np.array(self.all_vis_bias), np.array(self.all_hid_bias)

    def energy_activation(self, vis, hid, class_num):
        '''
        hid_bias 1 x 1000
        vis_bias 1 x 784

        vis 980 x 784
        hid 980 x 1000
        weights 784 x 1000

        '''
        vis_bias = self.all_vis_bias[class_num]
        hid_bias = self.all_hid_bias[class_num]
        weights = self.all_weights[class_num]

        return - np.dot(vis, vis_bias.T) - np.dot(hid, hid_bias.T) - np.dot(np.dot(hid, weights.T), vis.T)

    def test_class(self, class_num):

        print('Loading mnist test set for class %d' % class_num)
        vis, n_cases = load_mnist(self.directory, class_num, 'test')
        print('Building model for test class %d' % class_num)
        hid = self.trainer.update_hidden(vis)[1]

        log_prob_all = np.zeros((self.num_classes, n_cases))

        for class_num in range(self.num_classes):

            activation = self.energy_activation(vis, hid, class_num)

            log_prob = - np.sum(np.log(softmax(activation)), axis=1)
            log_prob_all[class_num] = log_prob

        print(log_prob_all[:,0])
        print(np.argmin(log_prob_all, axis=0))





if __name__ == "__main__":

    directory = '../data/mnist'
    num_classes = 10
    num_epochs = 5
    batchsize = 1000
    temps = np.arange(0.9, 1, 0.025)
    rmse_array = []
    filename = 'mnist_all.mat'
    sampler_type = 'hamiltonian'

    args = sys.argv[1:]
    if len(args) > 1:
        sampler_type = args[1]

    # convert_to_text(filename)

    test_model = TestMNIST(directory, num_classes)
    test_model.train_all(num_epochs, batchsize, sampler_type)
    test_model.test_class(0)

    # ====== MNIST DATA SET =======
    # print('Beginning to load mnist data')
    # data = load_mnist(directory, 0)
    # print('Finished loading mnist data')

    # print('Initializing RBM')
    # rbm = RBM(784, 1000)
    # rbm.params[:] = np.random.uniform(-1./10, 1./10, len(rbm.params))
    #
    # print('Training RBM')
    # trainer = CDTrainer(rbm)
    #
    # for i in range(len(temps)):
    #     rmse = trainer.train(data, num_epochs, batchsize=batchsize, temp=temps[i], sampler=sampler_type)
    #     rmse_array.append(rmse)
    #
    # epochs = np.arange(num_epochs)
    #
    # plt.figure(1)
    # plt.plot(epochs, rmse_array[0], 'ro', epochs, rmse_array[1], 'bs', epochs,
    #          rmse_array[2], 'g^', epochs, rmse_array[3], 'r--')
    # plt.xlabel('Epochs')
    # plt.ylabel('RMSE')
    # plt.savefig('LL_tempered_gibbs.png')
    # plt.show()

    # vis_all_classes, hid_all_classes = train_all(directory, num_epochs, batchsize)
