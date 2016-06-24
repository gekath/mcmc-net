import numpy as np
import theano.tensor

class HMC():

    def __init__(self, pos, mom, step_size, mass):

        self.pos = pos
        self.mom = mom
        self.step = step_size
        self.mass = mass

        self.pos_dim = pos.shape
        self.mom_dim = mom.shape

        try:
            assert self.pos_dim == self.mom_dim
        except AssertionError:
            print('Position and momentum vectors should have same dimension')


def simulate(init_pos, init_mom, step_size, num_steps, energy_fn):

    def leapfrog(leap_pos, leap_mom, step):

        mass = 1  # assume for now

        # calculate dU / d pos
        pos_grad = theano.tensor.grad(energy_fn(leap_pos).sum(), leap_pos)

        # calculate mom( t + step_size/2)
        mom_half = leap_mom - (step / 2) * pos_grad

        # calculate pos(t + step_size)
        new_pos = leap_pos + (step / mass) * mom_half

        return new_pos, mom_half

    # Calc energy at initial position, and half step of momentum
    # from init_pos i.e. pos(t_0)
    init_energy = energy_fn(init_pos)
    init_pos_grad = theano.tensor.grad(init_energy.sum(), init_pos)

    # momentum half step at time: t + step_size//2
    mom = init_mom - 0.5 * step_size * init_pos_grad

    # calculate position at time: t + step_size
    pos = init_pos + step_size * mom

    i = 0
    while i < num_steps:
        pos, mom = leapfrog(pos, mom, step_size)

    final_energy = energy_fn(pos)
    final_pos = pos
    final_pos_grad = theano.tensor.grad(final_energy.sum(), final_pos)

    # Calculate final momentum
    final_mom = mom - 0.5 * step_size * final_pos_grad

    return final_pos, final_mom


def hamiltonian(pos, mom, energy_fn):

    return energy_fn(pos) + kinetic(mom)


def kinetic(mom):
    '''
    Return kinetic energy (assume mass = 1)

    :param mom A momentum vector

    '''
    return 0.5 * (mom ** 2).sum(axis=1)


def metropolis_hastings_accept(num_generator, energy_prev, energy_next):

    energy_diff = energy_next - energy_prev
    dim = energy_prev.shape
    return (theano.tensor.exp(energy_diff) - num_generator.uniform(dim)) >= 0


def hmc_move(num_generator, position, step_size, num_steps, energy_fn):

    # assume univariate Gaussian for momentum
    pos_dim = position.shape
    init_mom = num_generator.normal(pos_dim)

    final_pos, final_mom = simulate(
        position,
        init_mom,
        step_size,
        num_steps,
        energy_fn)

    energy_prev = hamiltonian(position, init_mom, energy_fn)
    energy_next = hamiltonian(final_pos, final_mom, energy_fn)

    accept = metropolis_hastings_accept(num_generator, energy_prev, energy_next)

    return final_pos, accept


def hmc_updates(positions, step_size, avg_accept_rate, final_pos, accept,
                target_accept_rate, step_size_inc, step_size_dec,
                step_size_min, step_size_max, avg_accept_slowness):


    # have accept matrix with same dimensions as final_pos
    accept_matrix = accept.dimshuffle(0, *(('x',) * (final_pos.ndim - 1)))
    new_positions = theano.tensor.switch(accept_matrix, final_pos, positions)

    mean_dtype = theano.scalar.upcast(accept.dtype, avg_accept_rate.dtype)
    new_accept_rate = theano.tensor.add(
        avg_accept_slowness * avg_accept_rate,
        (1.0 - avg_accept_slowness) * accept.mean(dtype=mean_dtype))

    

# theano random number generator
num_generator = theano.tensor.shared_randomstreams.RandomStreams()

#
# def leapfrog(pos, mom, step_size, mass, energy_fn):
#
#     pos_grad = theano.tensor.grad(energy_fn(pos).sum(), pos)
#
#     mom_half = mom - (step_size / 2) * np.dot(pos_grad, pos)
#     new_pos = pos + (step_size / mass) * mom_half
#     new_mom = mom_half - (step_size / 2) * np.dot(pos_grad, new_pos)
#
#
#     return new_pos, new_mom

