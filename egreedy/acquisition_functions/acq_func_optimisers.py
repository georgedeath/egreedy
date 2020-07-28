"""A set of classes that take in a GPy model and optimise their respective
acquisition functions over the model's decision space.

Each class can be used as follows:
>> acq_class = EI
>> acq_optimiser = acq_class(lb, ub, acq_budget, cf=None, args)
>> acq_optimiser(gpy_model)

The basic usage is that an optimiser is instantiated with the problem bounds,
``lb`` and ``ub``, a budget of calls to the GPy model (used for predicting the
mean and variance of locations in decision space), a constraint function that
returns True or False depending on if the decision vector it is given violates
any problem constraints, and additional arguments in the form of a dictionary
containing key: value pairs that are passed into the acquisition function used
by the optimiser; e.g. for the UCB acquisition function the value of beta is
needed and can be specified: args = {'beta': 2.5}.

Note that all acquisition optimisers use the NSGA-II algorithm apart from PI
which uses DIRECT.
"""
import nlopt
import numpy as np

from . import standard_acq_funcs_minimize
from . import egreedy_acq_funcs_minimize
from .nsga2_pareto_front import NSGA2_pygmo


class BaseOptimiser:
    """Class of methods that maximise an acquisition function over a GPy model.

    Parameters
    ----------
        lb : (D, ) numpy.ndarray
            Lower bound box constraint on D
        ub : (D, ) numpy.ndarray
            Upper bound box constraint on D
        acq_budget : int
            Maximum number of calls to the GPy model
        cf : callable, optional
            Constraint function that returns True if it is called with a
            valid decision vector, else False.
        acquisition_args : dict, optional
            A dictionary containing key: value pairs that will be passed to the
            corresponding acquisition function, see the classes below for
            further details.
    """

    def __init__(self, lb, ub, acq_budget, cf=None, acquisition_args={}):
        self.lb = lb
        self.ub = ub
        self.cf = cf
        self.acquisition_args = acquisition_args
        self.acq_budget = acq_budget

    def __call__(self, model):
        raise NotImplementedError()


class ParetoFrontOptimiser(BaseOptimiser):
    """Class of acquisition function optimisers that use Pareto fronts.

    The (estimated) Pareto front is calculated using NSGA-II [1]_, for full
    details of the method see: nsga2_pareto_front.NSGA2_pygmo

    References
    ----------
    .. [1] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan.
       A fast and elitist multiobjective genetic algorithm: NSGA-II.
       IEEE Transactions on Evolutionary Computation 6, 2 (2001), 182–197.
    """

    def get_front(self, model):
        """Gets the (estimated) Pareto front of the predicted mean and
        standard deviation of a GPy.models.GPRegression model.
        """
        X_front, musigma_front = NSGA2_pygmo(
            model, self.acq_budget, self.lb, self.ub, self.cf
        )

        return X_front, musigma_front[:, 0], musigma_front[:, 1]

    def __call__(self, model):
        raise NotImplementedError()


class EI(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front that maximises EI.

    See standard_acq_funcs_minimize.EI for details of the EI method.
    """

    def __call__(self, model):
        X, mu, sigma = self.get_front(model)
        ei = standard_acq_funcs_minimize.EI(mu, sigma, y_best=np.min(model.Y))
        return X[np.argmax(ei), :]


class UCB(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front that maximises UCB.

    See standard_acq_funcs_minimize.UCB for details of the UCB method and its
    optional arguments.
    """

    def __call__(self, model):
        X, mu, sigma = self.get_front(model)
        ucb = standard_acq_funcs_minimize.UCB(
            mu,
            sigma,
            lb=self.lb,
            ub=self.ub,
            t=model.X.shape[0] + 1,
            d=model.X.shape[1],
            **self.acquisition_args
        )
        return X[np.argmax(ucb), :]


class eFront(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front via the eFront method.

    eFront greedily selects a point an (estimated) Pareto front that has the
    best (lowest) mean predicted value with probability (1 - epsilon) and
    randomly selects a point on the front with probability epsilon.
    """

    def __call__(self, model):
        X, mu, sigma = self.get_front(model)
        Xnew = egreedy_acq_funcs_minimize.eFront(X, mu, sigma, **self.acquisition_args)
        return Xnew


class eRandom(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front via the eRandom method.

    eRandom greedily selects a point an (estimated) Pareto front that has the
    best (lowest) mean predicted value with probability (1 - epsilon) and
    randomly selects a point in decision space with probability epsilon.
    """

    def __call__(self, model):
        X, mu, sigma = self.get_front(model)
        Xnew = egreedy_acq_funcs_minimize.eRandom(
            X, mu, sigma, lb=self.lb, ub=self.ub, cf=self.cf, **self.acquisition_args
        )
        return Xnew


class PFRandom(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front via the PFRandom method.

    PFRandom randomly selects a point on the Pareto front.
    """

    def __call__(self, model):
        X, _, _ = self.get_front(model)
        Xnew = egreedy_acq_funcs_minimize.PFRandom(X)
        return Xnew


class Explore(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front via the Explore method.

    Explore selects the most exploratory point on the front, i.e. the location
    with the largest standard deviation.
    """

    def __call__(self, model):
        X, _, sigma = self.get_front(model)
        Xnew = egreedy_acq_funcs_minimize.Explore(X, sigma)
        return Xnew


class Exploit(ParetoFrontOptimiser):
    """Selects the point on a GPy model's Pareto front via the Exploit method.

    Exploit selects the most exploitative point on the front, i.e. the location
    with the best (lowest) mean predicted value.
    """

    def __call__(self, model):
        X, mu, _ = self.get_front(model)
        Xnew = egreedy_acq_funcs_minimize.Exploit(X, mu)
        return Xnew


class PI(BaseOptimiser):
    """Maximises PI acquisition function for a given GPy model.

    See standard_acq_funcs_minimize.PI for details of the PI method.

    Notes
    -----
    PI is maximised using DIRECT because the point that maximises PI is not
    guaranteed to be on the Pareto front - see the paper for full details.
    """

    def __call__(self, model):
        D = model.X.shape[1]
        incumbent = model.Y.min()

        def min_obj(x, grad):
            # if we have a constraint function and it is violated,
            # return a bad PI value
            if (self.cf is not None) and (not self.cf(x)):
                return np.inf

            mu, sigmaSQR = model.predict(np.atleast_2d(x), full_cov=False)

            # negate PI as DIRECT minimises so we want the largest negative PI
            r = -standard_acq_funcs_minimize.PI(mu, np.sqrt(sigmaSQR), incumbent).flat[
                0
            ]
            return r

        # define a direct optimisation instance
        opt = nlopt.opt(nlopt.GN_DIRECT_L_RAND, D)
        opt.set_min_objective(min_obj)

        # problem bounds
        opt.set_lower_bounds(self.lb)
        opt.set_upper_bounds(self.ub)

        # budget and function tolerance
        opt.set_maxeval(self.acq_budget)
        opt.set_ftol_abs(1e-6)

        # initial random location
        x0 = np.random.uniform(low=self.lb, high=self.ub)

        # run DIRECT
        xopt = opt.optimize(x0)

        return xopt
