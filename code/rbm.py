import numpy as np

class RBM():

    def __init__(self, data, vis_dim=784, hid_dim=500):

        self.data = data
        self.vis_dim = vis_dim
        self.hid_dim = hid_dim

        # TODO: initialize W
        self.W = np.zeros((vis_dim, hid_dim))

        #TODO: initialize b == v_bias
        self.v_bias = np.zeros(vis_dim)

        # TODO: initialize c == h_bias
        self.h_bias = np.zeros(hid_dim)

        self.params = [self.W, self.v_bias, self.h_bias]


    def positive_phase(self):





    def free_energy(self, sample):

        wx_b = np.dot(sample, self.W) + self.h_bias
        e_wx_b = np.dot(self.v_bias, wx_b)
        hidden_term = np.sum(np.log(1 + e_wx_b))
        visible_term = np.dot(sample, self.v_bias)

        return - visible_term - hidden_term

