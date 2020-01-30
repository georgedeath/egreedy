"""This module contains functions allowing for the generation of training data
for each of the test problems and also includes the functions required to
generate the additional arguments, specifying the target locations, of the
robot pushing tasks (push4 and push8).
"""
from pyDOE2 import lhs

import numpy as np
from .. import test_problems


def generate_training_data_LHS(problem_name, n_exp_start=1, n_exp_end=51,
                               N_samples=None, optional_arguments={}):
    """Generates training data for a given problem with (optional) arguments.

    This function generates training data for a specific test problem using
    Latin hypercube sampling such that the minimum distance between samples has
    been maximised. If there are function arguments specified then these are
    passed into the function during its instantiation and before samples are
    generated.

    The data is saved to the 'data' directory in a file named:
    '{problem_name}_{exp_no}.npz' and contains two items, 'arr_0' corresponding
    to the N_samples locations evaluated and has shape (N_samples, D), 'arr_1'
    contains the results of each function evaluation and has shape
    (N_samples, 1), and an optional third item, 'arr_2' containing a dictionary
    of key:value pairs passed to the function that generated the corresponding
    training data.

    Parameters
    ----------
    problem_name : string
        Problem name, must exist as a class imported from the test_problems
        module.
    n_exp_start : int
        Starting number of the experiment.
    n_exp_end : int
        Ending number (inclusive) of the experiment.
    N_samples : int, optional
        Number of Latin hypercube samples of the function to carry out. This
        defaults to twice the dimensionality of the function's decision space.
    optional_arguments : dict
        Dictionary of optional arguments. For each key-value pair of the
        dictionary, the key should correspond to an argument needed for the
        function and the value should be a numpy.ndarray of length
        ``n_exp_end`` - ``n_exp_start`` + 1; i.e. each element of the array
        will be used as a value in the function. See the example below for more
        details.

    Examples
    --------
    # for a test problem , WangFreitas, with no function arguments:
    >>> generate_training_data_LHS('WangFreitas', n_exp_start=1, n_exp_end=3)
    Saved: WangFreitas_1.npz
    Saved: WangFreitas_2.npz
    Saved: WangFreitas_3.npz
    #
    # for a test problem, push4, with two function arguments:
    >>> T1_x, T1_y = generate_push4_targets(3)
    >>> push4_targets = {'t1_x': T1_x, 't1_y': T1_y}
    >>> print(push4_targets)
    {'t1_x': array([ 2.88191356,  1.06383215, -4.48898253]),
     't1_y': array([ 0.5626891 ,  3.81250785, -4.49494566])}
    >>> generate_training_data_LHS('push4', 1, 3,
                                   optional_arguments=push4_targets)
    Saved: push4_1.npz
    Saved: push4_2.npz
    Saved: push4_3.npz
    """
    exp_nos = np.arange(n_exp_start, n_exp_end + 1)
    N = len(exp_nos)

    # check that there are the same number of arguments as there are
    # experimental training data to construct
    for _, v in optional_arguments.items():
        assert len(v) == N, "There should be as many elements for each " \
                            "optional arguments as there are experimental runs"

    # get the function class
    f_class = getattr(test_problems, problem_name)

    for i, exp_no in enumerate(exp_nos):
        # get the optional arguments for this problem instance (if they exist)
        opt_args = {k: v[i] for (k, v) in optional_arguments.items()}

        # instantiate the function
        f = f_class(**opt_args)

        # if N_samples isn't specified, generate 2 * D samples
        N_samples = 2 * f.dim if N_samples is None else N_samples

        # LHS and rescale to decision space
        Xtr = (f.ub - f.lb) * lhs(f.dim, N_samples, criterion='maximin') + f.lb

        # evaluate the function and reshape to (N_samples, 1)
        Ytr = np.reshape(f(Xtr), (-1, 1))

        # save the results
        fn = f'training_data/{problem_name:s}_{exp_no:d}.npz'
        np.savez(fn, Xtr, Ytr, opt_args)
        print('Saved: {:s}'.format(fn))


def generate_training_data_PitzDaily(n_exp_start=1, n_exp_end=51,
                                     N_samples=20):
    """Generates training data the PitzDaily test problem.

    Samples are generated uniformly, rather than with LHS, because the
    PitzDaily test problem has a constraint function.

    Parameters
    ----------
    n_exp_start : int
        Starting number of the experiment.
    n_exp_end : int
        Ending number (inclusive) of the experiment.
    N_samples : int, optional
        Number of Latin hypercube samples of the function to carry out. This
        defaults 20 (twice the problem dimensionality).

    Examples
    --------
    >>> # generate experimental runs 1 to 51 (inclusive) - note this will take
    >>> # approximately 17 hours (51 * 20  minutes)
    >>> generate_training_data_PitzDaily()
    Generating data: PitzDaily_1.npz
        Finished 1/20
        Finished 2/20
        ...
        Finished 20/20
    Saved: PitzDaily_1.npz
    ...
    Saved: PitzDaily_51.npz
    """
    exp_nos = np.arange(n_exp_start, n_exp_end + 1)

    # instantiate the test problem
    f = test_problems.PitzDaily()

    # number of samples to generate - default is 2 * D
    N_samples = 2 * f.dim if N_samples is None else N_samples

    for exp_no in exp_nos:
        Xtr = np.zeros((N_samples, 10))
        Ytr = np.zeros((N_samples, 1))

        fn = f'training_data/PitzDaily_{exp_no:d}.npz'
        print('Generating data: {:s}'.format(fn))

        # generate and evaluate solutions that do not violate the constraint
        # function
        for i in range(N_samples):
            Xtr[i] = f.generate_valid_solution()
            Ytr[i] = f(Xtr[i])
            print('\tFinished {:d}/{:d}'.format(i + 1, N_samples))

        np.savez(fn, Xtr, Ytr, {})
        print('Saved: {:s}\n'.format(fn))


def generate_push4_targets(N):
    """Generates target locations for the 'push4' problems by LHS.

    Generates ``N`` target locations, defined as [x, y] pairs each in [-5, 5],
    using Latin hypercube sampling.

    Parameters
    ----------
    N : int
        Number of locations to generate.

    Returns
    -------
    T1_x : (N, ) numpy.ndarray
        x-axis location of the targets.
    T1_y : (N, ) numpy.ndarray
        y-axis locations of the targets.
    """
    # lower and upper bounds
    T_lb = np.array([-5, -5])
    T_ub = np.array([5, 5])

    # LHS sample and rescale from [0, 1]^2 to the bounds above
    T = lhs(2, N, criterion='maximin') * (T_ub - T_lb) + T_lb

    T1_x, T1_y = T.T
    return T1_x, T1_y


def generate_push8_targets(N):
    """Generates pairs of target locations for the two robots in push8.

    The function generates two set of ``N`` LHS samples and pairs up the
    locations such that the distance between them is >= 1.1, thereby giving
    room for the objects (with diameter = 1) to each sit perfectly on their
    target without blocking one another.

    Parameters
    ----------
    N : int
        Number of pairs of locations to generate.

    Returns
    -------
    T1_x : (N, ) numpy.ndarray
        x-axis location of the first targets in the pairs
    T1_y : (N, ) numpy.ndarray
        y-axis locations of the first targets in the pairs
    T2_x : (N, ) numpy.ndarray
        x-axis location of the second targets in the pairs
    T2_y : (N, ) numpy.ndarray
        y-axis locations of the second targets in the pairs
    """
    # we can call generate_push4_targets(N) twice to get two sets of locations
    T1 = np.concatenate([generate_push4_targets(N)]).T
    T2 = np.concatenate([generate_push4_targets(N)]).T

    # ensure the pair of targets is greater than 1.1 away from each other by
    # shuffling the samples to be paired together - thus not allowing objects
    # to overlap with each other if they are perfectly positioned on the target
    while True:
        norm = np.linalg.norm([T1 - T2], axis=1)
        if np.min(norm) >= 1.1:
            break

        np.random.shuffle(T2)

    T1_x, T1_y = T1.T
    T2_x, T2_y = T2.T

    return T1_x, T1_y, T2_x, T2_y


if __name__ == "__main__":
    # how to generate push4 training data:
    T1_x, T1_y = generate_push4_targets(51)
    push4_targets = {'t1_x': T1_x, 't1_y': T1_y}
    generate_training_data_LHS('push4', 1, 51,
                               optional_arguments=push4_targets)

    # how to generate push8 training data:
    T1_x, T1_y, T2_x, T2_y = generate_push8_targets(51)
    push8_targets = {'t1_x': T1_x, 't1_y': T1_y,
                     't2_x': T2_x, 't2_y': T2_y}
    generate_training_data_LHS('push8', 1, 51,
                               optional_arguments=push8_targets)

    # how to generate generate data for the synthetic test problems
    generate_training_data_LHS('WangFreitas', 1, 51)
