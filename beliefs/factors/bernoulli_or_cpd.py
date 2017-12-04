import numpy as np

from beliefs.factors.cpd import TabularCPD


class BernoulliOrCPD(TabularCPD):
    """CPD class for a Bernoulli random variable whose relationship to its
    parents (also Bernoulli random variables) is described by OR logic.

    If at least one of the variable's parents is True, then the variable
    is True, and False otherwise.
    """
    def __init__(self, variable, parents=[]):
        """
        Args:
          variable: int or string
          parents: optional, list of int and/or strings
        """
        super().__init__(variable=variable,
                         variable_card=2,
                         parents=parents,
                         parents_card=[2]*len(parents),
                         values=[])
        self._values = []

    @property
    def values(self):
        if not any(self._values):
            self._values = self._build_kwise_values_array(len(self.variables))
            self._values = self._values.reshape(self.cardinality)
        return self._values

    @staticmethod
    def _build_kwise_values_array(k):
        # special case a completely independent factor, and
        # return the uniform prior
        if k == 1:
            return np.array([0.5, 0.5])

        return np.array(
            [1.,] + [0.]*(2**(k-1)-1) + [0.,] + [1.]*(2**(k-1)-1)
        )