"""This module contains methods to randomly sample, either uniformly or using
LHS, test problems for a fix budget (inclusive of the initial training data).
"""
import numpy as np
from pyDOE2 import lhs
from .. import test_problems


def perform_LHS_runs(problem_name, budget=250, n_exp_start=1, n_exp_end=51):
    """Generates the LHS samples for a fixed budget (inclusive of training).

    Generates a fixed budget of LHS locations (inclusive of those in the
    training data) for optimisation runs in [n_exp_start, n_exp_end]. This
    function is not for the PitzDaily test problem because it has a
    constraint function and therefore needs to be uniformly sampled from to
    make sure each sample generated does not violate the constraint; instead,
    ``perform_uniform_runs`` can be used.

    The results of the will be stored in the 'results' directory with the name:
    '{problem_name:}_{run_no:}_{budget:}_LHS.npz'

    Parameters
    ----------
    problem_name : string
        Test problem name to perform the optimisation run on. This will
        attempt to import problem_name from the test_problems module.
    budget : int
        Total number of expensive evaluations to carry out. Note that
        the budget includes points in the initial training data.
    n_exp_start : int
        Starting number of the experiment.
    n_exp_end : int
        Ending number (inclusive) of the experiment.
    """
    exp_nos = np.arange(n_exp_start, n_exp_end + 1)

    # get the function class
    f_class = getattr(test_problems, problem_name)

    for exp_no in exp_nos:
        # paths to data
        data_file = f"training_data/{problem_name:}_{exp_no:}.npz"
        save_file = f"results/{problem_name:}_{exp_no:}_{budget:}_LHS"

        # load the initial LHS locations
        with np.load(data_file, allow_pickle=True) as data:
            Xtr = data["arr_0"]
            Ytr = data["arr_1"]
            if "arr_2" in data:
                f_optional_arguments = data["arr_2"].item()
            else:
                f_optional_arguments = {}

        # instantiate the function
        f = f_class(**f_optional_arguments)
        f_lb = f.lb
        f_ub = f.ub
        f_dim = f.dim

        # samples to generate
        N_samples = budget - Xtr.shape[0]

        # LHS, rescale to decision space and evaluate
        _X = (f_ub - f_lb) * lhs(f_dim, N_samples, criterion="maximin") + f_lb
        _Y = np.reshape(f(_X), (N_samples, 1))

        # combine with the training data and save
        Xtr = np.concatenate((Xtr, np.atleast_2d(_X)))
        Ytr = np.concatenate((Ytr, np.atleast_2d(_Y)))
        np.savez(save_file, Xtr=Xtr, Ytr=Ytr)

        print("Generated samples: {:s}".format(save_file))


def perform_uniform_runs(problem_name, budget=250, n_exp_start=1, n_exp_end=51):
    """Generates the uniformly distributed samples for a fixed budget.

    Generates a fixed budget of uniformly distributed samples (inclusive of
    the number of training points training). A uniform set of samples should be
    generated for problems that have constraints (i.e. non-box constraints);
    the PitzDaily test problem is an example of this. This function will
    repeatedly generate uniform samples until it has a sufficient number of
    them that do not violate the test problem's constraint function.

    The results of the will be stored in the 'results' directory with the name:
    '{problem_name:}_{run_no:}_{budget:}_UNIFORM.npz'

    Parameters
    ----------
    problem_name : string
        Test problem name to perform the optimisation run on. This will
        attempt to import problem_name from the test_problems module. The
        test problem should have a callable constraint function once it is
        instantiated named 'cf' that takes in a decision vector and returns a
        True if the decision vector does not violate the constraint function.
    budget : int
        Total number of expensive evaluations to carry out. Note that
        the budget includes points in the initial training data.
    n_exp_start : int
        Starting number of the experiment.
    n_exp_end : int
        Ending number (inclusive) of the experiment.
    """
    exp_nos = np.arange(n_exp_start, n_exp_end + 1)

    # get the function class
    f_class = getattr(test_problems, problem_name)

    for exp_no in exp_nos:
        # paths to data
        data_file = f"training_data/{problem_name:}_{exp_no:}.npz"
        save_file = f"results/{problem_name:}_{exp_no:}_{budget:}_UNIFORM"

        # load the initial training data locations
        with np.load(data_file, allow_pickle=True) as data:
            Xtr = data["arr_0"]
            Ytr = data["arr_1"]
            if "arr_2" in data:
                f_optional_arguments = data["arr_2"].item()
            else:
                f_optional_arguments = {}

        # instantiate the function
        f = f_class(**f_optional_arguments)
        f_lb = f.lb
        f_ub = f.ub
        f_dim = f.dim
        f_cf = f.cf

        # samples to generate
        N_samples = budget - Xtr.shape[0]

        Xnew = np.zeros((N_samples, f_dim))
        Ynew = np.zeros((N_samples, 1))

        for i in range(N_samples):
            # uniformly sample decision vectors until one passes the cf
            while True:
                x = np.random.uniform(f_lb, f_ub)
                if f_cf is None or f_cf(x):
                    break

            # expensively evaluate
            Xnew[i] = x
            Ynew[i] = f(x)

        # combine with the training data and save
        Xtr = np.concatenate((Xtr, np.atleast_2d(Xnew)))
        Ytr = np.concatenate((Ytr, np.atleast_2d(Ynew)))
        np.savez(save_file, Xtr=Xtr, Ytr=Ytr)

        print("Generated samples: {:s}".format(save_file))
