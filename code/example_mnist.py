from rbm4 import *
from utils import *
import numpy as np
import sys
import scipy.io as sio
from sklearn.linear_model import LogisticRegression


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
        self.model = None

    def train_all(self, num_epochs, hmc_params, batchsize, cd_steps, num_hid=500):

        self.all_vis_bias = []
        self.all_hid_bias = []
        self.all_weights = []

        for class_num in range(self.num_classes):

            print('Beginning to load MNIST dataset ' + str(class_num))
            data, n_cases = load_mnist(self.directory, class_num)
            print('Finished loading MNIST dataset ' + str(class_num))

            print('Initializing RBM' + str(class_num))
            self.model = RBM(784, num_hid)
            # rbm.params[:] = np.random.uniform(-1./10, 1./10, len(rbm.params))

            print('Training RBM' + str(class_num))
            self.trainer = Trainer(self.model)

            params = self.trainer.train(data, num_epochs,
                                              hmc_params,
                                              batchsize=batchsize,
                                              cd_steps=cd_steps
                                              )
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

        vis_term = - np.dot(vis, vis_bias.T).flatten()
        hid_term = - np.dot(hid, hid_bias.T).flatten()
        weight_term = - np.dot(np.dot(hid, weights.T), vis.T)[0]

        return vis_term + hid_term + weight_term

    def free_energy(self, vis, class_num):
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

        vis_term = np.dot(vis, vis_bias.T).flatten()
        hid_term = np.dot(vis, weights) + hid_bias.flatten()
        weights_term = np.log(1 + np.exp(hid_term))
        sum_weights_term = np.sum(weights_term, axis=1)
        final_free_energy = - vis_term - sum_weights_term
        return final_free_energy

    def predict(self, class_num, temp=1):

        print('Loading mnist test set for class %d' % class_num)
        vis, n_cases = load_mnist(self.directory, class_num, 'test')
        print('Building model for test class %d' % class_num)
        activation, hid = self.model.update_hidden(vis)

        log_prob_all = np.zeros((self.num_classes, n_cases))

        for num in range(self.num_classes):

            prob = self.free_energy(vis, num) / temp

            log_prob = - np.log(softmax(prob))
            log_prob_all[num] = log_prob

        # print(log_prob_all[:,0])
        print(log_prob_all[0])
        predictions = np.argmax(log_prob_all, axis=0)
        # print(predictions)
        # print(predictions == class_num)

        error = np.sum(predictions == class_num) / float(n_cases)
        print('Error %f ' % error)

        return error

if __name__ == "__main__":

    directory = '../data/mnist'
    num_classes = 3
    num_epochs = 25
    batchsize = 100
    temps = np.arange(0.8, 1.2, 0.05)
    error_temps = []
    filename = 'mnist_all.mat'
    sampler_type = 'Gibbs'
    cd_steps = 1
    predict_class_num = 2
    num_hid = 500

    # Hamiltonian params
    hmc_params = {  'num_steps': 20,
                    'step_size': 0.0001,
                    'num_iters': 10,
                    'mass': 1,
                    'temp': 1,
                    'step_size_adj': 0.001
                 }

    args = sys.argv[1:]
    if len(args) > 1:
        sampler_type = args[1]

    # convert_to_text(filename)

    test_model = TestMNIST(directory, num_classes)
    test_model.train_all(num_epochs, hmc_params, batchsize, cd_steps, num_hid)

    for i in range(len(temps)):

        print('Temperature of %f' % temps[i])
        error = test_model.predict(predict_class_num, temps[i])
        error_temps.append(error)

    print('Error rate')
    print(error_temps)

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
