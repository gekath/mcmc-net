import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt

class HMCSampler():

    def __init__(self, params, energy_fn, grad_fn):

        self.step_size = params['step_size']
        self.num_steps = params['num_steps']
        self.num_iters = params['num_iters']
        self.step_size_adj = params['step_size_adj']

        self.mass = params['mass']
        self.temp = params['temp']

        self.energy_fn = energy_fn
        self.grad_fn = grad_fn

    def kinetic_energy(self, mom):
        '''
        :param mom: Momentum vector of dimension d
        :param mass: Matrix of dimension d x d (if None, mass is identity matrix)
        :return: kinetic energy given momentum vector
                    Assume for now, mass = identity of dimension mom
        '''
        return 0.5 * np.dot(mom.T, mom) / self.mass

    def potential_energy(self, pos):
        '''
        :param pos: Momentum vector of dimension d
        :return: Negative log probability given position vector
        '''
        # TODO: Calculate log likelihood function

        # sampling from salpeter mass function
        pos = pos[0]
        c = (1.0 - pos)/(math.pow(M_max, 1.0-pos) - math.pow(M_min, 1.0-pos))
        return - (N*math.log(c) - pos*D)

    def calc_gradient(self, pos):
        # TODO: calculate gradient of log likelihood

        # sampling from salpeter mass function
        pos = pos[0]
        grad = logMmin*math.pow(M_min, 1.0-pos) - logMmax*math.pow(M_max, 1.0-pos)
        grad = 1.0 + grad*(1.0 - pos)/(math.pow(M_max, 1.0-pos)
                                         -math.pow(M_min, 1.0-pos))
        grad = -D - N*grad/(1.0 - pos)
        return np.array(grad)

    def mom_update(self, pos, mom):
        grad = - self.grad_fn(pos)
        return mom - 0.5 * self.step_size * grad


    def pos_update(self, pos, mom):
        return pos + self.step_size * mom


    def leapfrog_updates(self, pos, mom):
        '''
        :param pos: Position vector
        :param mom: Momentum vector
        :param step_size:
        :return: The new position and momentum vectors after one full step.
                    Assume mass is the identity matrix
        '''

        # momentum half-step
        mom_half = self.mom_update(pos, mom)

        # position full step
        pos_new = self.pos_update(pos, mom_half)

        # momentum full step
        mom_new = self.mom_update(pos_new, mom_half)

        return pos_new, mom_new

    def hmc_move(self, init_state=[1]):

        accepted = 0.0
        chain = [init_state]
        mom_chain = [[0]]
        dim = init_state.shape[1]

        for iters in range(self.num_iters):

            # Last state stored in chain
            old_pos = chain[len(chain) - 1]
            # Assume momentum is univariate Gaussian
            mom = np.random.normal(0.0, 1.0, dim)
            old_hamiltonian = self.kinetic_energy(mom) + self.energy_fn(old_pos)
            old_grad = - self.grad_fn(old_pos)

            new_pos = copy.copy(old_pos)

            for i in range(self.num_steps):
                new_pos, mom = self.leapfrog_updates(new_pos, mom)

            new_energy = self.energy_fn(new_pos)
            new_hamiltonian = new_energy + self.kinetic_energy(mom)
            energy_diff = new_hamiltonian - old_hamiltonian

            if accept_state(energy_diff):
                chain.append(new_pos)
                mom_chain.append(mom)
                accepted = accepted + 1.0
            else:
                chain.append(old_pos)
                mom_chain.append(mom)

            acceptance_rate = accepted/float(len(chain))

            # TODO: update step size dynamically to maintain certain acceptance
            if acceptance_rate > 0.9:
                self.step_size += 0.00001
                print(self.step_size)
            elif acceptance_rate < 0.6:
                self.step_size -= 0.00001

        # acceptance_rate = accepted / float(len(chain))

        print("Acceptance rate = " + str(acceptance_rate))
        # return np.array(chain), np.array(mom_chain)
        # Confabulations accepted are the final state of the chain
        return chain[-1]


def accept_state(energy_diff):
    '''
    :param energy_diff: Change in Hamiltonian energy
    :return: True iff energy_diff < 0, or exp(energy_diff) greater than uniform random num
    '''

    if np.min(energy_diff) < 0.0:
        return True
    else:
        u = random.uniform(0.0, 1.0)
        return u < math.exp(-np.min(energy_diff))


if __name__ == "__main__":

    chain, mom_chain = run_hmc(potential_energy, calc_gradient, [3])

    clean = []
    clean_mom = []

    # for i in range(len(chain)):
    #     clean.append(chain[i][0])
    #     clean_mom.append(mom_chain[i][0])

    for n in range(len(chain) / 2, len(chain)):
        if (n % 10 == 0):
            clean.append(chain[n][0])
            # clean_mom.append(mom_chain[n])

    print(clean)
    print(clean_mom)

    print("Mean:" + str(np.mean(clean)))
    print("Sigma:" + str(np.std(clean)))

    plt.figure(1)
    plt.hist(clean, 20, histtype='step', lw=3)
    plt.xticks([2.346, 2.348, 2.35, 2.352, 2.354],
               [2.346, 2.348, 2.35, 2.352, 2.354])
    plt.xlim(2.345, 2.355)
    plt.xlabel(r'$\alpha$', fontsize=24)
    plt.ylabel(r'$\cal L($Data$;\alpha)$', fontsize=24)
    plt.savefig('example-MCMC-results2.png')
    plt.show()

    # plt.figure(1)
    # plt.plot(clean, clean_mom, 'ro')
    # plt.xlabel('Position')
    # plt.ylabel('Momentum')
    # plt.savefig('example-mcmc-circle2.png')
    # plt.show()