import numpy as np


class TabularCPD:
    """
    Defines the conditional probability table for a discrete variable
    whose parents are also discrete.

    TODO: have this inherit from DiscreteFactor implementing explicit factor methods
    """
    def __init__(self, variable, variable_card,
                 parents=[], parents_card=[], values=[]):
        """
        Args:
          variable: int or string
          variable_card: int
          parents: optional, list of int and/or strings
          parents_card: optional, list of int
          values: optional, 2d list or array
        """
        self.variable = variable
        self.parents = parents
        self.variables = [variable] + parents
        self.cardinality = [variable_card] + parents_card
        self._values = np.array(values)

    @property
    def values(self):
        return self._values

    def get_values(self):
        """
        Returns the tabular cpd form of the values.
        """
        if len(self.cardinality) == 1:
            return self.values.reshape(1, np.prod(self.cardinality))
        else:
            return self.values.reshape(self.cardinality[0], np.prod(self.cardinality[1:]))

    def copy(self):
        return self.__class__(self.variable,
                              self.cardinality[0],
                              self.parents,
                              self.cardinality[1:],
                              self._values)
