"""PitzDaily: A computationally expensive ten-dimensional CFD simulation.

For more details, see the PitzDaily class and its associated references.
"""
import os
import sys
import tempfile
import numpy as np


class PitzDaily:
    """PitzDaily computational fluid dynamics problem.

    PitzDaily CFD problem by [1]_. This is a 10D test problem that involves
    optimising the shape of a pipe in order to reduce the pressure loss between
    inflow and outflow. For full details of the problem see [1]_. This test
    problem makes use of the CDF test suite assocated with [1]_:
    https://bitbucket.org/arahat/cfd-test-problem-suite/

    .. [1] Steven J. Daniels, Alma A. M. Rahat, Richard M. Everson,
       Gavin R. Tabor, and Jonathan E. Fieldsend. 2018.
       A Suite of Computationally Expensive Shape Optimisation Problems Using
       Computational Fluid Dynamics.
       In Parallel Problem Solving from Nature – PPSN XV. Springer, 296–307.
    """
    def __init__(self):
        self.dim = 10
        self.yopt = 0
        self.prob = None
        self.problem_lb = None
        self.problem_ub = None
        self.problem_cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        val = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            with PitzDailyProblem() as prob:
                # this has to be called to set the problem up
                prob.get_decision_boundary()
                val[i] = prob.evaluate(x[i, :], verbose=False)

        return val.ravel()

    # lazy instantiation of constraint function and bounds
    def _set_problem_attributes(self):
        if self.prob is None:
            test_instance = PitzDailyProblem()
            self.prob = test_instance.getprob()
            self.problem_lb, self.problem_ub = self.prob.get_decision_boundary()
            self.problem_cf = self.prob.constraint

    def generate_valid_solution(self, max_attempts=1000):
        """Attempts to randomly generate a valid PitzDaily solution.

        Parameters
        ----------
        prob : .cfd_test_problem_suite.Exeter_CFD_Problems.PitzDaily instance
            PitzDaily problem instance from the CFD suite.
        max_attempts : int
            Maximum number of attempts to try and generate a valid solution.
        """
        self._set_problem_attributes()

        for _ in range(max_attempts):
            x = np.random.uniform(self.lb, self.ub)

            if self.cf(x):
                return x

        raise Exception('No valid solution could be generated')

    @property
    def lb(self):
        self._set_problem_attributes()
        return self.problem_lb

    @property
    def ub(self):
        self._set_problem_attributes()
        return self.problem_ub

    @property
    def cf(self):
        self._set_problem_attributes()
        return self.problem_cf


class PitzDailyProblem(object):
    """PitzDaily problem object.

    This class allows the use of the with statement to both instantiate and
    clean up the temporary files used in the CFD computation.

    Examples
    --------
    >>> with PitzDailyProblem() as prob:
            # generate a valid solution to evaluate
            x = generate_valid_solution(prob)
            result = prob.evaluate(x, verbose=False)
    >>> print(result)
    0.612872766622
    """
    def __init__(self):
        # removing all command line arguments because openFOAM tries to read
        # these in even though they are given to a different python script
        sys.argv = sys.argv[:1]

        # directory containing the problem info
        ddir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                            'Exeter_CFD_Problems/data/'))

        # create a temporary directory for the CFD run
        self.temp_dir_object = tempfile.TemporaryDirectory()
        self.temp_dir_path = self.temp_dir_object.name + '/'

        # settings for the CFD computation
        self.settings = {
            'source_case': os.path.join(ddir, 'PitzDaily/case_fine/'),
            'case_path': self.temp_dir_path,
            'boundary_files': [os.path.join(ddir, 'PitzDaily/boundary.csv')],
            'fixed_points_files': [os.path.join(ddir, 'PitzDaily/fixed.csv')]
        }

        # importing here so we only use the CFD module if explicitly called
        from . import Exeter_CFD_Problems

        self.prob = Exeter_CFD_Problems.PitzDaily(self.settings)

    def getprob(self):
        return self.prob

    def __del__(self):
        self.temp_dir_object.cleanup()

    def __enter__(self):
        return self.getprob()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.temp_dir_object.cleanup()


if __name__ == "__main__":
    # Check the test problem can be called (i.e. test OpenFOAM)

    # instantiate the test problem
    f = PitzDaily()
    print('PitzDaily successfully instantiated..')

    # generate a valid solution
    x = f.generate_valid_solution()
    print('Generated valid solution, evaluating..')

    # evaluate
    fx = f(x)
    print('Fitness value:', fx)
