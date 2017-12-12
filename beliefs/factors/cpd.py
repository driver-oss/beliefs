import numpy as np
from beliefs.factors.discrete_factor import DiscreteFactor


class TabularCPD(DiscreteFactor):
    """
    Defines the conditional probability table for a discrete variable
    whose parents are also discrete.
    """
    def __init__(self, variable, variable_card,
                 parents=[], parents_card=[], values=[], state_names=None):
        """
        Args:
          variable: int or string
          variable_card: int
          parents: optional, list of int and/or strings
          parents_card: optional, list of int
          values: optional, 2d list or array
          state_names: dictionary (optional),
                mapping variables to their states, of format {label_name: ['state1', 'state2']}
        """
        super().__init__(variables=[variable] + parents,
                         cardinality=[variable_card] + parents_card,
                         values=values,
                         state_names=state_names)
        self.variable = variable
        self.parents = parents

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
