"""Acquisition functions that use the (estimated) Pareto front (PF):

eFront: Greedily selects the best the location on the (given estimated) PF that
        has the best (smallest) value with probability (1-epsilon) and the rest
        of the time randomly selects a location on the PF.

eRandom: Similar to eFront but performs its random selection across the entire
         decision space.

PFRandom: Randomly selects a point on the PF; equivalent to eFront with
          epsilon = 1.

Exploit. Greedily selects the location on the PF with the best predicted mean
         value; equivalent to eFront and eRandom with epsilon = 0.

Explore: Greedily selects the location on the PF with the largest predicted
         variance.
"""
import numpy as np


def eFront(X, mu, sigma, epsilon=0.1):
    """Perform e-greedy selection (eFront) on a Pareto front of mu and sigma.

    Greedily selects the minimum (best) mean value (``mu``) 1-epsilon of the
    time and selects randomly from the Pareto front the remaining time.

    Parameters
    ----------
        X : (N, D) numpy.ndarray
            The locations, in decision space, on the Pareto front.
        mu : (N, ) numpy.ndarray
            Mean value predictions of the elements on the Pareto front.
        sigma : (N, ) numpy.ndarray
            Standard deviation of the elements on the Pareto front.
        epsilon : float in [0, 1]
            Proportion of times to select an element on the front at random.

    Returns
    -------
        Xnew : (D, ) numpy.ndarray
            New location to expensively evaluate.
    """
    # randomly select, with chance epsilon, a location on the Pareto front
    if np.random.rand() < epsilon:
        return X[np.random.choice(mu.size), :]

    # else greedily, with chance (1-epsilon), select the location with the
    # best mean function value
    return X[np.argmin(mu.ravel()), :]


def eRandom(X, mu, sigma, lb, ub, cf=None, epsilon=0.1):
    """Perform e-greedy selection (eRandom) on a Pareto front of mu and sigma.

    Greedily selects the minimum (best) mean value (``mu``) 1-epsilon of the
    time and selects from the entire feasible space (1-epsilon) of the time.

    Parameters
    ----------
        X : (N, D) numpy.ndarray
            N locations, in D-dimension decision space, on the Pareto front.
        mu : (N, ) numpy.ndarray
            Mean value predictions of the elements on the Pareto front.
        sigma : (N, ) numpy.ndarray
            Standard deviation of the elements on the Pareto front.
        lb : (D, ) numpy.ndarray
            Lower bound box constraint on D
        ub : (D, ) numpy.ndarray
            Upper bound box constraint on D
        cf : callable, optional
            Constraint function that returns True if it is called with a
            valid decision vector, else False.
        epsilon : float in [0, 1]
            Proportion of times to select an element in the entire decision
            space, as defined by the problem's lower and upper bounds
            ``lb`` and ``ub`` respectively.

    Returns
    -------
        Xnew : (D, ) numpy.ndarray
            New location to expensively evaluate.
    """
    # randomly select, with chance epsilon, a location in X
    if np.random.rand() < epsilon:
        Xnew = np.random.uniform(lb, ub)

        # if we have a constraint function, keep generating random solutions
        # in decision space until we get a valid one
        if cf is not None:
            while not cf(Xnew):
                Xnew = np.random.uniform(lb, ub)

        return Xnew

    # else greedily, with chance (1-epsilon), select the location with the
    # best mean function value
    return X[np.argmin(mu.ravel()), :]


def PFRandom(X):
    """Randomly selects a location on the Pareto front.

    Returns an element randomly selected from X.

    Parameters
    ----------
        X : (N, D) numpy.ndarray
            N locations, in D-dimension decision space, on the Pareto front.

    Returns
    -------
        Xnew : (D, ) numpy.ndarray
            New location to expensively evaluate.
    """
    return X[np.random.choice(X.shape[0]), :]


def Explore(X, sigma):
    """Returns the location on the Pareto front with the highest standard deviation.

    Parameters
    ----------
        X : (N, D) numpy.ndarray
            The locations, in decision space, on the Pareto front.
        mu : (N, ) numpy.ndarray
            Mean value predictions of the elements on the Pareto front.
        sigma : (N, ) numpy.ndarray
            Standard deviation of the elements on the Pareto front.

    Returns
    -------
        Xnew : (D, ) numpy.ndarray
            New location to expensively evaluate.
    """
    return X[np.argmax(sigma.ravel()), :]


def Exploit(X, mu):
    """Returns the location on the Pareto front with the best (lowest) mean prediction.

    Parameters
    ----------
        X : (N, D) numpy.ndarray
            The locations, in decision space, on the Pareto front.
        mu : (N, ) numpy.ndarray
            Mean value predictions of the elements on the Pareto front.
        sigma : (N, ) numpy.ndarray
            Standard deviation of the elements on the Pareto front.

    Returns
    -------
        Xnew : (D, ) numpy.ndarray
            New location to expensively evaluate.
    """
    return X[np.argmin(mu.ravel()), :]
