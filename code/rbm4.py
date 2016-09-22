import numpy as np
import time
from hamiltonian import *
import PIL.Image as Image
from utils import *

class RBM():
    '''
    Energy based model with n_vis visible units and n_hid hidden units.
    '''

    def __init__(self, n_vis, n_hid):

        self.n_vis = n_vis
        self.n_hid = n_hid

        # weight matrix of dimension n_hid x n_vis
        # initialized with uniform distribution
        self.weights = np.random.uniform(-1./10, 1./10, (n_vis, n_hid))

        # vis bias vector of dimension n_vis
        self.visbias = np.zeros((1, n_vis))

        # hid bias vector of dimension n_hid
        self.hidbias = np.zeros((1, n_hid))

    def update_hidden(self, vis):
        '''
        activation = W_i v + c_i
        Return activation, and sigmoid(activation)
        '''

        prod_vis_weights = np.dot(vis, self.weights)
        activation = prod_vis_weights + self.hidbias

        return activation, 1 / ( 1 + np.exp( - activation))

    def update_visible(self, hid):
        '''
        activation = h' W_j + v_j
        Return activation, and sigmoid(activation)
        '''

        prod_hid_weights = np.dot(hid, self.weights.T)
        activation = prod_hid_weights + self.visbias

        return activation, 1 / (1 + np.exp( - activation))

    def free_energy(self, vis):

        vis_term = np.dot(vis, self.visbias.T)
        activation = np.dot(vis, self.weights) + self.hidbias
        sigmoid = np.log(1 + np.exp(activation))
        sum_sigmoid = np.sum(sigmoid, axis=1)
        final_free_energy = - vis_term - sum_sigmoid

        return np.sum(final_free_energy)


class Trainer():
    '''
    Uses contrastive divergence with an MCMC sampler to train given model
    over given data.
    '''

    def __init__(self, model, weight_cost=0.0002, rates=(1e-4, 1e-4, 1e-4)):

        self.model = model
        self.weight_cost = weight_cost
        self.vis_rate, self.hid_rate, self.weight_rate = rates
        self.weight_step = np.zeros(model.weights.shape)

    def gibbs_sampler(self, num_steps, h0_sample):
        '''
        Perform Gibbs sampling for num_steps steps, given a vector of hidden units.
        '''
        for step in range(num_steps):

            v1_activation, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
            h1_activation, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return v1_sample, h1_sample

    # def sample_h_given_v(self, v0_sample):
    #
    #     h1_activation, h1_mean = self.model.update_hidden(v0_sample)
    #     h1_sample = h1_activation
    #     # h1_sample = np.random.binomial(1, h1_mean, h1_mean.shape)
    #     return h1_activation, h1_mean, h1_sample
    #
    # def sample_v_given_h(self, h0_sample):
    #
    #     v1_activation, v1_mean = self.model.update_visible(h0_sample)
    #     v1_sample = v1_activation
    #     # v1_sample = np.random.binomial(1, v1_mean, v1_mean.shape)
    #     return v1_activation, v1_mean, v1_sample

    def sample(self, probs):

        samples = np.random.uniform(size=probs.shape)
        probs[samples < probs] = 1.
        np.floor(probs, probs)

    def sample_h_from_v(self, vis):

        h_dim = self.model.n_hid
        batchsize, n_dim = vis.shape

        hid = self.model.update_hidden(vis)[1]
        pos_state = hid > np.random.rand(batchsize, h_dim)

        return hid, pos_state

    def train(self, data, num_epochs, hmc_params, batchsize=100, cd_steps=1, momentum=0.9):
        '''
        Train model given data using specified sampling method by updating
        weights for num_epochs number of epochs.
        '''

        # number of cases, dimension of data

        n_cases, n_dim = data.shape
        h_dim = self.model.n_hid

        try:

            for epoch in range(num_epochs):

                epoch_start = time.clock()

                for offset in range(0, n_cases, batchsize):

                    batch = data[offset:offset+batchsize]
                    if batch.shape[0] != batchsize:
                        break

                    batch = batch > np.random.rand(batchsize, n_dim)
                    # print(batch[0])
                    # print(batch.shape)

                    # Initialize hidden layer, given the current batch data
                    # Should be dimension n_cases x n_hid
                    hid = self.model.update_hidden(batch)[1]

                    # Positive phase
                    # hid, pos_states = self.sample_h_from_v(batch)

                    pre_weight = np.dot(batch.T, hid)
                    pre_hid_sum = np.sum(hid, axis=0)
                    pre_vis_sum = np.sum(batch, axis=0)

                    # vis, hid = self.hamiltonian(hid, batch)
                    # vis, hid = self.gibbs_sampler(cd_steps, hid)

                    pos_states = hid > np.random.rand(batchsize, h_dim)
                    #
                    # # Negative phase
                    #
                    # vis = self.model.update_visible(pos_states)[1]
                    # vis = vis > np.random.rand(batchsize, n_dim)
                    #
                    # hid = self.model.update_hidden(vis)[1]

                    vis, hid = self.hamiltonian(pos_states, batch, hmc_params)
                    hid = hid > np.random.rand(batchsize, h_dim)

                    post_weight = np.dot(vis.T, hid)
                    post_hid_sum = np.sum(hid, axis=0)
                    post_vis_sum = np.sum(vis, axis=0)

                    # Update weights
                    diff_model_weights = (pre_weight - post_weight) / batchsize
                    weight_product = self.model.weights * self.weight_cost
                    weight_step = self.weight_rate*(diff_model_weights - weight_product)

                    self.weight_step = self.weight_step * momentum
                    self.weight_step += weight_step
                    self.model.weights += self.weight_step

                    # Update vis bias
                    vis_step = pre_vis_sum - post_vis_sum
                    vis_step = vis_step * self.vis_rate / batchsize
                    self.model.visbias += vis_step

                    # Update hid bias
                    hid_step = pre_hid_sum - post_hid_sum
                    hid_step = hid_step * self.hid_rate / batchsize
                    self.model.hidbias += hid_step

                    # Calculate MSE (not good measure for this case, though will
                    # give an approximate idea of correct direction of training)
                    mse = np.sum((vis - batch) ** 2) / n_cases

                rmse = np.sqrt(mse)
                print "Epoch %d: %f seconds, RMSE=%f" % \
                      (epoch + 1, time.clock() - epoch_start, rmse)
                print('Constructing image')
                image = Image.fromarray(tile_raster_images(X=self.model.weights.T,
                                                           img_shape=(28, 28),
                                                           tile_shape=(10, 10),
                                                           tile_spacing=(1, 1)))
                image.save('rbm_filter_epoch_%i.png' % epoch)
                print('Finished constructing image')

        except KeyboardInterrupt:
            print('Keyboard Interrupt')

        finally:
            return (self.model.weights, self.model.visbias, self.model.hidbias)

    def gradient(self, vis):
        activation = np.dot(vis, self.model.weights) + self.model.hidbias
        sigmoid = 1 / (1 + np.exp(- activation))
        final_gradient = np.dot(sigmoid.T, vis)

        return np.sum(final_gradient, axis=0)


    def hamiltonian(self, hid, init_state, params):
        # hid = np.atleast_2d(hid)
        # ncases = hid.shape[0]

        hmc = HMCSampler(params, self.model.free_energy, self.gradient)
        new_vis = hmc.hmc_move(init_state)
        new_hid = self.model.update_hidden(new_vis)[1]

        return new_vis, new_hid
