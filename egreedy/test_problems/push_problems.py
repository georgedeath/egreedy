"""Real-world robot pushing problems: push4 and push8.

push4 is the "4-D ACTION" problem from [1]_, a four-dimensional robot pushing
task in which a robot has to push and object to an unknown target and receives
feedback, after completing its task, in the form of the distance from the
pushed object to the target. The robot is parametrised by its initial
location, the angle of its rectangular hand (used for pushing) and the number
of time-steps it pushes for.

push8 is an extension of the push4 problem in which there are two robots
pushing to targets to unknown target locations. The robots can block each
other and therefore the problem will likely be much harder than the push4
problem.

Each problem class, once instantiated can be directed called, eg:
>>> f = push4(t1_x=4, t1_y=4)
>>> x = numpy.array([-4, -4, 100, (1 / 4) * numpy.pi])
>>> f(x)
array([3.75049018])

Additional parameters can also be specified to allow the visualisation and
saving of the robot pushing problem. Assuming the function as be instantiated
as above, a dictionary named "plotting_args" can be passed during the function
call, e.g. f(x, plotting_args) to facilitate plotting. Its keys are as follows:
    'show': bool
        Whether to show the pushing problem.
    'save': bool
        Whether to save the shown images - also needs 'show' to be True.
    'save_dir': str
        Directory to save the images to. It must exist and be set if saving.
    'save_every': int
        N'th frame to be saved; e.g. if set to 10 every 10th frame gets saved.
    'save_prefix': str
        Prefix given to the saved filenames - useful when saving multiple runs.

Note that once initialised (details in each problem definition) the problem
domain for each problem is mapped to [0, 1]^D because the time-step parameter
(in [0, 300]) is an order of magnitude larger than the initial robot position
(in [-5, 5]) and hand orientation [0, 2*numpy.pi].

References
----------
.. [1] Zi Wang and Stefanie Jegelka. 2017.
   Max-value entropy search for efficient Bayesian optimization.
   In Proceedings of the 34th International Conference on Machine Learning.
   PMLR, 3627–3635.
"""
import numpy as np
from .push_world import push_4D, push_8D


class push4:
    """Robot pushing simulation: One robot pushing one object.

    The object's initial location is at [0, 0] and the robot travels in the
    direction of the object's initial position. See paper for full problem
    details.

    Parameters
    ----------
    tx_1 : float
        x-axis location of the target, should reside in [-5, 5].
    ty_1 : float
        y-axis location of the target, should reside in [-5, 5].

    Examples
    --------
    >> f_class = push4
    >> # set initial target location (unknown to the robot and only
    >> # used for distance calculation after it has finished pushing)
    >> tx_1 = 3.5; ty_1 = 4
    >> # instantiate the test problem
    >> f = f_class(tx_1, ty_1)
    >> # evaluate some solution x in [0, 1]^4
    >> x = numpy.array([0.5, 0.7, 0.2, 0.3])
    >> f(x)
    array([9.0733461])
    """
    def __init__(self, t1_x=0, t1_y=0):
        self.dim = 4
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        self.lb_orig = np.array([-5.0, -5.0, 1.0, 0])
        self.ub_orig = np.array([5.0, 5.0, 300.0, 2.0 * np.pi])

        # object target location
        self.t1_x = t1_x
        self.t1_y = t1_y

        # initial object location (0,0) as in Wang et al. (see module comment)
        self.o1_x = 0
        self.o1_y = 0

        # optimum location unknown as defined by inputs
        self.yopt = 0

        self.cf = None

    def __call__(self, x, plotting_args=None):
        x = np.atleast_2d(x)

        # map from unit space to original space
        x = x * (self.ub_orig - self.lb_orig) + self.lb_orig

        val = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            val[i, :] = push_4D(x[i, :],
                                self.t1_x, self.t1_y,
                                self.o1_x, self.o1_y,
                                plotting_args)

        return val.ravel()


class push8:
    """Robot pushing simulation: Two robots pushing an object each.

    The objects' initial locations are at [-3, 0] and [3, 0] respectively,
    with the robot 1 pushing the first target and robot 2 pushing the second
    target. See paper for full problem details.

    Parameters
    ----------
    tx_1 : float
        x-axis location for the target of robot 1, should reside in [-5, 5].
    ty_1 : float
        y-axis location for the target of robot 1, should reside in [-5, 5].
    tx_2 : float
        x-axis location for the target of robot 2, should reside in [-5, 5].
    ty_2 : float
        y-axis location for the target of robot 2, should reside in [-5, 5].

    Examples
    --------
    >> f_class = push8
    >> # initial positions (tx_1, ty_1) and (tx_2, ty_2) for both robots
    >> tx_1 = 3.5; ty_1 = 4
    >> tx_2 = -2; ty_2 = 1.5
    >> # instantiate the test problem
    >> f = f_class(tx_1, ty_1, tx_2, ty_2)
    >> # evaluate some solution x in [0, 1]^8
    >> x = numpy.array([0.5, 0.7, 0.2, 0.3, 0.3, 0.1, 0.5, 0.6])
    >> f(x)
    array([24.15719287])
    """
    def __init__(self, t1_x=-5, t1_y=-5, t2_x=5, t2_y=5):
        self.dim = 8
        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        self.lb_orig = np.array([-5.0, -5.0, 1.0, 0,
                                 -5.0, -5.0, 1.0, 0])
        self.ub_orig = np.array([5.0, 5.0, 300.0, 2 * np.pi,
                                 5.0, 5.0, 300.0, 2 * np.pi])

        # object target locations
        self.t1_x = t1_x
        self.t1_y = t1_y
        self.t2_x = t2_x
        self.t2_y = t2_y

        # initial object locations (-3, 0) and (3, 0)
        self.o1_x = -3
        self.o1_y = 0
        self.o2_x = 3
        self.o2_y = 0

        # optimum location unknown as defined by inputs
        self.yopt = 0

        self.cf = None

    def __call__(self, x, plotting_args=None):
        x = np.atleast_2d(x)

        # map from unit space to original space
        x = x * (self.ub_orig - self.lb_orig) + self.lb_orig

        val = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            val[i, :] = push_8D(x[i, :],
                                self.t1_x, self.t1_y,
                                self.t2_x, self.t2_y,
                                self.o1_x, self.o1_y,
                                self.o2_x, self.o2_y,
                                plotting_args)

        return val.ravel()
