"""Three common acquisition functions (also known as infill criteria),
Expected Improvement (EI), Probability of Improvement (PI) and Upper
Confidence Bound (UCB).
"""
import numpy as np
from scipy.stats import norm


def EI(mu, sigma, y_best, PHI=norm.cdf, phi=norm.pdf):
    """The Expected Improvement (EI) over the best observed value.

    This acquisition function calculates the EI over the current best
    observed value ``y_best`` for given inputs, defined by their predicted
    mean ``mu`` and standard deviation ``sigma``. EI was first proposed in
    Močkus et al. [1]_ and later developed by Jones et al. [2]_.

    Parameters
    ----------
    mu : (N, ) numpy.ndarray
        Mean predicted value for N locations.
    sigma : (N, ) numpy.ndarray
        Predicted standard deviation for the corresponding N locations.
    y_best : float
        Best observed value.

    Returns
    -------
    ei : (N, ) numpy.ndarray
        The Expected improvement of the N locations.

    Notes
    -----
    Expected Improvement for minimisation.

    References
    ----------
    .. [1] Jonas Močkus, Vytautas Tiešis, and Antanas Žilinskas. 1978.
       The application of Bayesian methods for seeking the extremum.
       Towards Global Optimization 2, 1 (1978), 117–129.

    .. [2] Donald R. Jones, Matthias Schonlau, and William J. Welch. 1998.
       Efficient Global Optimization of Expensive Black-Box Functions.
       Journal of Global Optimization 13, 4 (1998), 455–492.

    """
    # ensure mu and sigma are column vectors
    mu = np.reshape(mu, (-1, 1))
    sigma = np.reshape(sigma, (-1, 1))

    # minimisation version of EI
    improvement = y_best - mu

    # EI = 0 if sigma = 0, so mask the non-zero sigma elements
    ei = np.zeros_like(improvement)
    mask = (sigma != 0).ravel()

    s = improvement[mask] / sigma[mask]
    ei[mask] = improvement[mask] * PHI(s) + sigma[mask] * phi(s)

    return ei


def PI(mu, sigma, y_best, phi=norm.pdf):
    """The Probability of Improvement (PI) over the best observed value.

    This acquisition function calculates the PI over the current best
    observed value ``y_best`` for given inputs, defined by their predicted
    mean ``mu`` and standard deviation ``sigma``.
    PI was first proposed by [1]_.

    Parameters
    ----------
    mu : (N, ) numpy.ndarray
        Mean predicted value for N locations.
    sigma : (N, ) numpy.ndarray
        Predicted standard deviation for the corresponding N locations.
    y_best : float
        Best observed value.

    Returns
    -------
    pi : (N, ) numpy.ndarray
        The Probability of Improvement of the N locations.

    Notes
    -----
    Probability of Improvement for minimisation.

    References
    ----------
    .. [1] Harold J. Kushner. 1964.
       A new method of locating the maximum point of an arbitrary multipeak
       curve in the presence of noise.
       Journal Basic Engineering 86, 1, 97–106.
    """
    # ensure mu and sigma are column vectors
    mu = np.reshape(mu, (-1, 1))
    sigma = np.reshape(sigma, (-1, 1))

    # minimisation version of PI
    improvement = y_best - mu

    # PI = 0 if sigma = 0, so mask the non-zero sigma elements
    pi = np.zeros_like(improvement)
    mask = (sigma != 0).ravel()

    s = improvement[mask] / sigma[mask]
    pi[mask] = phi(s)

    return pi


def UCB(mu, sigma, lb, ub, t, d, beta="theorem2", sizeD=1000, a=1, b=1, B=10):
    """The Upper Confidence Bound (UCB) acquisition function.

    First proposed by Lai and Robbins [1]_, with convergence proofs given
    by Srinivas et al. [2]_, the UCB function is simply the weighted sum of
    the mean prediction ``mu`` and standard deviation ``sigma``, where the
    weight, ``beta`` controls the trade-off between exploitation and
    exploration:

    .. math:: UCB(mu, sigma) = mu + sqrt(beta) sigma

    Larger values of beta give a greater preference to exploratory locations
    whereas smaller values give a greater preference towards more
    exploitative locations.

    This function implements four possible uses for beta, with descriptions
    of the relevant theorems available in [2].
    - beta='theorem1' calculates beta in the formula in theorem 1 of [2]_, and
                      requires the arguments ``sizeD`` and ``t`` to be set.
    - beta='theorem2' calculates beta in the formula in theorem 2 of [2]_.
                      requires the arguments ``lb``, ``ub``, ``d``, ``a``,
                      ``b` and ``t`` to be set.
    - beta='theorem3' calculates beta in the formula in theorem 3 of [2]_ and
                      requires the arguments ``B``, ``t`` and ``d`` to be set.
    - beta=float uses the value of beta specified.

    Parameters
    ----------
    mu : (N, ) numpy.ndarray
        Mean predicted value for N locations.
    sigma : (N, ) numpy.ndarray
        Predicted standard deviation for the corresponding N locations.
    y_best : float
        Best observed value.

    Returns
    -------
    ucb : (N, ) numpy.ndarray
        The UCB calculcated for a given beta for the N locations.

    Notes
    -----
    Technically the function can be classed as the Lower confidence bound of
    the mean minus the variance - although to match the literature we refer to
    this as the upper bound.

    References
    ----------
    .. [1] Tze Leung Lai and Herbert Robbins. 1985.
       Asymptotically efficient adaptive allocation rules.
       Advances in Applied Mathematics 6, 1 (1985), 4–22.

    .. [2] Niranjan Srinivas, Andreas Krause, Sham Kakade, and Matthias Seeger.
       2010. Gaussian process optimization in the bandit setting: no regret and
       experimental design. In Proceedings of the 27th International Conference
       on Machine Learning. Omnipress, 1015–1022.
    """
    delta = 0.01  # as specified in section 6 of [2].

    if beta == "theorem1":
        # Theorem 1 - Bounded and finite D
        beta = 2 * np.log(sizeD * t ** 2 + np.pi ** 2 / (6 * delta))

    elif beta == "theorem2":
        # Theorem 2 - Bounded and infinite D - f sampled from GP
        r = np.max(ub - lb)

        beta = 2 * np.log(t ** 2 * 2 * np.pi ** 2 / (3 * delta)) + 2 * d * np.log(
            t ** 2 * d * b * r * np.sqrt(np.log(4 * d * a / delta))
        )
    elif beta == "theorem3":
        # Theorem 3 - Bounded and infinite D - f bounded RKHS
        gamma = np.log(t) ** (d + 1)
        beta = 2 * B + 300 * gamma * np.log(t / delta) ** 3

    # else we use the given beta value
    return -(mu - np.sqrt(beta) * sigma)
