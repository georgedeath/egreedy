"""NSGA-II function for estimating the Pareto front of the GPy model.

"""
import numpy as np
import pygmo as pg


def NSGA2_pygmo(model, fevals, lb, ub, cf=None):
    """Finds the estimated Pareto front of a GPy model using NSGA2 [1]_.

    Parameters
    ----------
    model : GPy.models.gp_regression.GPRegression
        GPy regression model on which to find the Pareto front of its mean
        prediction and standard deviation.
    fevals : int
        Maximum number of times to evaluate a location using the model.
    lb : (D, ) numpy.ndarray
        Lower bound box constraint on D
    ub : (D, ) numpy.ndarray
        Upper bound box constraint on D
    cf : callable, optional
        Constraint function that returns True if it is called with a
        valid decision vector, else False.

    Returns
    -------
    X_front : (F, D) numpy.ndarray
        The F D-dimensional locations on the estimated Pareto front.
    musigma_front : (F, 2) numpy.ndarray
        The corresponding mean response and standard deviation of the locations
        on the front such that a point X_front[i, :] has a mean prediction
        musigma_front[i, 0] and standard deviation musigma_front[i, 1].

    Notes
    -----
    NSGA2 [1]_ discards locations on the pareto front if the size of the front
    is greater than that of the population size. We counteract this by storing
    every location and its corresponding mean and standard deviation and
    calculate the Pareto front from this - thereby making the most of every
    GP model evaluation.

    References
    ----------
    .. [1] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan.
       A fast and elitist multiobjective genetic algorithm: NSGA-II.
       IEEE Transactions on Evolutionary Computation 6, 2 (2001), 182–197.
    """
    # internal class for the pygmo optimiser
    class GPY_WRAPPER(object):
        def __init__(self, model, lb, ub, cf, evals):
            # model = GPy model
            # lb = np.array of lower bounds on X
            # ub = np.array of upper bounds on X
            # cf = callable constraint function
            # evals = total evaluations to be carried out
            self.model = model
            self.lb = lb
            self.ub = ub
            self.nd = lb.size
            self.got_cf = cf is not None
            self.cf = cf
            self.i = 0  # evaluation pointer

        def get_bounds(self):
            return (self.lb, self.ub)

        def get_nobj(self):
            return 2

        def fitness(self, X):
            X = np.atleast_2d(X)
            f = model_fitness(X, self.model, self.cf, self.got_cf,
                              self.i, self.i + X.shape[0])
            self.i += X.shape[0]
            return f

    # fitness function for the optimiser
    def model_fitness(X, model, cf, got_cf, start_slice, end_slice):
        valid = True

        # if we select a location that violates the constraint,
        # ensure it cannot dominate anything by having its fitness values
        # maximally bad (i.e. set to infinity)
        if got_cf:
            if not cf(X):
                f = [np.inf, np.inf]
                valid = False

        if valid:
            mu, sigmaSQR = model.predict(X, full_cov=False)
            # note the negative sigmaSQR here as NSGA2 is minimising
            # so we want to minimise the negative variance
            f = [mu.flat[0], -np.sqrt(sigmaSQR).flat[0]]

        # store every point ever evaluated
        model_fitness.X[start_slice:end_slice, :] = X
        model_fitness.Y[start_slice:end_slice, :] = f

        return f

    # get the problem dimensionality
    D = lb.size

    # NSGA-II settings
    POPSIZE = D * 100
    N_GENS = int(np.ceil(fevals / POPSIZE))
    TOTAL_EVALUATIONS = POPSIZE * N_GENS

    nsga2 = pg.algorithm(pg.nsga2(gen=1,
                                  cr=0.8,       # cross-over probability.
                                  eta_c=20.0,   # distribution index (cr)
                                  m=1 / D,        # mutation rate
                                  eta_m=20.0))  # distribution index (m)

    # preallocate the storage of every location and fitness to be evaluated
    model_fitness.X = np.zeros((TOTAL_EVALUATIONS, D))
    model_fitness.Y = np.zeros((TOTAL_EVALUATIONS, 2))

    # problem instance
    gpy_problem = GPY_WRAPPER(model, lb, ub, cf, TOTAL_EVALUATIONS)
    problem = pg.problem(gpy_problem)

    # initialise the population
    population = pg.population(problem, size=POPSIZE)

    # evolve the population
    for i in range(N_GENS):
        population = nsga2.evolve(population)

    # indices non-dominated points across the entire NSGA-II run
    front_inds = pg.non_dominated_front_2d(model_fitness.Y)

    X_front = model_fitness.X[front_inds, :]
    musigma_front = model_fitness.Y[front_inds, :]

    # convert the standard deviations back to positive values; nsga2 minimises
    # the negative standard deviation (i.e. maximises the standard deviation)
    musigma_front[:, 1] *= -1

    return X_front, musigma_front
