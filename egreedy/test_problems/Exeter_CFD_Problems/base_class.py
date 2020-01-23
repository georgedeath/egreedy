# This is a skeleton for the Exeter CFD problems package.
# python version: 3

#####################
# Base class
#####################
from abc import ABCMeta, abstractmethod

class Problem(metaclass=ABCMeta):
    """
    The base class for CFD based functions.
    """

    @abstractmethod
    def info(self):
        """
        Show information about the problem.
        """
        pass

    @abstractmethod
    def get_configurable_settings(self):
        """
        Show configurable settings for the problem.
        """
        pass

    @abstractmethod
    def setup(self):
        """
        Set up for CFD simulation.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Run CFD simulation.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Collate results from CFD simulations and compute the objective function(s).
        """
        pass


class Interface(metaclass=ABCMeta):
    """
    An interface between shape parameters and decision vectors.
    """

    @abstractmethod
    def constraint(self):
        """
        Implement a constraint. This should be cheap to compute and return a boolean.
        """
        pass

    @abstractmethod
    def get_decision_boundary(self):
        """
        The primary decision space should be a hyperrectangle, and this method should
        return the lower bounds and upper bounds of the decision space.
        """
        pass

    @abstractmethod
    def convert_decision_to_shape(self):
        """
        Convert a decision vector to appropriate shape parameters.
        """
        pass

    @abstractmethod
    def convert_shape_to_decision(self):
        """
        Convert a set of shape parameters to decision vector. This is more to test
        if the conversion works both ways.
        """
        pass
'''
    @abstractmethod
    def cost_function(self):
        """
        Set up the CFD simulation and collate the results. Return single or
        multi-objective function values.
        """
        pass
'''
