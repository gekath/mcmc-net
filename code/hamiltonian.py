import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt

def kinetic_energy(mom, mass=None):
    '''
    :param mom: Momentum vector of dimension d
    :param mass: Matrix of dimension d x d (if None, mass is identity matrix)
    :return: kinetic energy given momentum vector
                Assume for now, mass = identity of dimension mom
    '''
    return 0.5 * np.dot(mom.T, mom)


def potential_energy(pos, temp=1):
    '''
    :param pos: Momentum vector of dimension d
    :return: Negative log probability given position vector
    '''
    # TODO: Calculate log likelihood function

    # sampling from salpeter mass function
    pos = pos[0]
    c = (1.0 - pos)/(math.pow(M_max, 1.0-pos) - math.pow(M_min, 1.0-pos))
    return - (N*math.log(c) - pos*D)

    # return 0.5 * np.dot(pos, pos)


def calc_gradient(pos, temp=1):
    # TODO: calculate gradient of log likelihood

    # sampling from salpeter mass function
    pos = pos[0]
    grad = logMmin*math.pow(M_min, 1.0-pos) - logMmax*math.pow(M_max, 1.0-pos)
    grad = 1.0 + grad*(1.0 - pos)/(math.pow(M_max, 1.0-pos)
                                     -math.pow(M_min, 1.0-pos))
    grad = -D - N*grad/(1.0 - pos)
    return np.array(grad)

    # return - np.array(pos)


def mom_update(pos, mom, step_size, gradient_function, temp):
    grad = - gradient_function(pos, temp)
    return mom - 0.5 * step_size * grad


def pos_update(pos, mom, step_size):
    return pos + step_size * mom


def leapfrog_updates(pos, mom, step_size, gradient_function, temp=1):
    '''
    :param pos: Position vector
    :param mom: Momentum vector
    :param step_size:
    :return: The new position and momentum vectors after one full step.
                Assume mass is the identity matrix
    '''

    # momentum half-step
    mom_half = mom_update(pos, mom, step_size, gradient_function, temp)

    # position full step
    pos_new = pos_update(pos, mom_half, step_size)

    # momentum full step
    mom_new = mom_update(pos_new, mom_half, step_size, gradient_function, temp)

    return pos_new, mom_new


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

#
# def sampleFromSalpeter(N, alpha, M_min, M_max):
#     # Convert limits from M to logM.
#     log_M_Min = math.log(M_min)
#     log_M_Max = math.log(M_max)
#     # Since Salpeter SMF decays, maximum likelihood occurs at M_min
#     maxlik = math.pow(M_min, 1.0 - alpha)
#
#     # Prepare array for output masses.
#     Masses = []
#     # Fill in array.
#     while (len(Masses) < N):
#         # Draw candidate from logM interval.
#         logM = random.uniform(log_M_Min,log_M_Max)
#         M    = math.exp(logM)
#         # Compute likelihood of candidate from Salpeter SMF.
#         likelihood = math.pow(M, 1.0 - alpha)
#         # Accept randomly.
#         u = random.uniform(0.0,maxlik)
#         if (u < likelihood):
#             Masses.append(M)
#     return Masses


def run_hmc(energy_function, gradient_function, init_state=[1], temp=1):

    #tODO: TUne parameres

    step_size = 0.000000047
    accepted = 0.0
    num_steps = 5
    chain = [init_state]
    mom_chain = [[0]]
    dim = init_state.shape[1]
    # dim = 1
    params = {"num_steps": num_steps,
              "step_size": step_size,
              "accepted": accepted,
              "init_state": init_state,
              }
    #TODO: Tune parameters

    for n in range(100):

        # Last state stored in chain
        old_pos = chain[len(chain) - 1]
        # Assume momentum is univariate Gaussian
        mom = np.random.normal(0.0, 1.0, dim)
        old_hamiltonian = kinetic_energy(mom) + energy_function(old_pos, temp)
        old_grad = - gradient_function(old_pos, temp)

        new_pos = copy.copy(old_pos)
        # new_grad = copy.copy(old_grad)

        for i in range(num_steps):
            new_pos, mom = leapfrog_updates(new_pos, mom, step_size, gradient_function, temp)
            # mom = mom - step_size*new_grad * 0.5
            # new_pos = new_pos + step_size * mom
            # new_grad = -calc_gradient(new_pos)
            # mom = mom - step_size*new_grad * 0.5
            chain.append(new_pos)
            mom_chain.append(mom)

        new_energy = energy_function(new_pos, temp)
        new_hamiltonian = new_energy + kinetic_energy(mom)
        energy_diff = new_hamiltonian - old_hamiltonian

        if accept_state(energy_diff):
            chain.append(new_pos)
            mom_chain.append(mom)
            accepted = accepted + 1.0
        else:
            chain.append(old_pos)
            mom_chain.append(mom)

    print("Acceptance rate = "+str(accepted/float(len(chain))))
    # return np.array(chain), np.array(mom_chain)
    # Confabulations accepted are the final state of the chain
    return chain[-1], mom_chain[-1]

# N = 1000000
# alpha = 2.35
# M_min = 1.0
# M_max = 100.0
# Masses = sampleFromSalpeter(N, alpha, M_min, M_max)
# LogM = np.log(np.array(Masses))
# D = np.mean(LogM) * N
# logMmin = math.log(1.0)
# logMmax = math.log(100.0)


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