import numpy as np
import time

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

    # def energy_function(self, vis, hid):
    #     return - np.dot(self.visbias, vis) - np.dot(self.hidbias, hid)


class Trainer():
    '''
    Uses contrastive divergence with an MCMC sampler to train given model
    over given data.
    '''

    def __init__(self, model, weight_cost=0.0002, rates=(1e-4, 1e-4, 1e-4)):

        self.model = model
        self.weight_cost = weight_cost
        self.vis_rate, self.hid_rate, self.weight_rate = rates

    def gibbs_sampler(self, num_steps, h0_sample):
        '''
        Perform Gibbs sampling for num_steps steps, given a vector of hidden units.
        '''
        for step in range(num_steps):

            v1_activation, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
            h1_activation, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)

        return v1_sample, h1_sample

    def update_hidden(self, vis):
        '''
        activation = W_i v + c_i
        Return activation, and sigmoid(activation)
        '''

        prod_vis_weights = np.dot(vis, self.model.weights)
        activation = prod_vis_weights + self.model.hidbias
        return activation, 1 / (1 + np.exp( - activation))

    def update_visible(self, hid):
        '''
        activation = h' W_j + v_j
        Return activation, and sigmoid(activation)
        '''

        prod_hid_weights = np.dot(hid, self.model.weights.T)
        activation = prod_hid_weights + self.model.visbias
        return activation, 1 / (1 + np.exp( - activation))

    def sample_h_given_v(self, v0_sample):

        h1_activation, h1_mean = self.update_hidden(v0_sample)
        h1_sample = h1_activation
        # h1_sample = np.random.binomial(1, h1_mean, h1_mean.shape)
        return h1_activation, h1_mean, h1_sample

    def sample_v_given_h(self, h0_sample):

        v1_activation, v1_mean = self.update_visible(h0_sample)
        v1_sample = v1_activation
        # v1_sample = np.random.binomial(1, v1_mean, v1_mean.shape)
        return v1_activation, v1_mean, v1_sample

    def train(self, data, num_epochs=50, batchsize=100, cd_steps=1, momentum=0.9, sampler='Gibbs'):
        '''
        Train model given data using specified sampling method by updating
        weights for num_epochs number of epochs.
        '''

        # number of cases, dimension of data
        n_cases, n_dim = data.shape
        h_dim = self.model.n_hid

        for epoch in range(num_epochs):

            epoch_start = time.clock()

            for offset in range(0, n_cases, batchsize):

                batch = data[offset:offset+batchsize]

                # Initialize hidden layer, given the current batch data
                # Should be dimension n_cases x n_hid
                hid = self.update_hidden(batch)[1]

                pre_hid_sum = np.sum(hid, axis=0)
                pre_weight = np.dot(batch.T, hid)

                vis, hid = self.gibbs_sampler(cd_steps, hid)

                # Update weights
                diff_model_weights = (pre_weight - np.dot(vis.T, hid)) / batchsize
                weight_product = self.model.weights * self.weight_cost
                weight_step = self.weight_rate * (diff_model_weights - weight_product)
                weight_step = momentum * weight_step
                self.model.weights += weight_step

                # Update vis bias
                vis_step = np.sum(batch, axis=0) - np.sum(vis, axis=0)
                vis_step = vis_step * self.vis_rate / batchsize
                self.model.visbias += vis_step

                # Update hid bias
                hid_step = pre_hid_sum - np.sum(hid, axis=0)
                hid_step = hid_step * self.hid_rate / batchsize
                self.model.hidbias += hid_step

                # Calculate MSE (not good measure for this case, though will
                # give an approximate idea of correct direction of training)
                mse = np.sum((vis - batch) ** 2) / n_cases

            rmse = np.sqrt(mse)
            print "Epoch %d: %f seconds, RMSE=%f" % \
                  (epoch + 1, time.clock() - epoch_start, rmse)

        return rmse, (self.model.weights, self.model.visbias, self.model.hidbias)
