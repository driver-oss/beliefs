import numpy as np


class BernoulliOrFactor:
    """CPD class for a Bernoulli random variable whose relationship to its
    parents is described by OR logic.

    If at least one of a child's parents is True, then the child is True, and
    False otherwise."""
    def __init__(self, child, parents=set()):
        self.child = child
        self.parents = set(parents)
        self.variables = set([child] + list(parents))
        self.cardinality = [2]*len(self.variables)
        self._values = None

    @property
    def values(self):
        if self._values is None:
            self._values = self._build_kwise_values_array(len(self.variables))
            self._values = self._values.reshape(self.cardinality)
        return self._values

    def get_values(self):
        """
        Returns the tabular cpd form of the values.
        """
        if len(self.cardinality) == 1:
            return self.values.reshape(1, np.prod(self.cardinality))
        else:
            return self.values.reshape(self.cardinality[0], np.prod(self.cardinality[1:]))

    @staticmethod
    def _build_kwise_values_array(k):
        # special case a completely independent factor, and
        # return the uniform prior
        if k == 1:
            return np.array([0.5, 0.5])

        return np.array(
            [1.,] + [0.]*(2**(k-1)-1) + [0.,] + [1.]*(2**(k-1)-1)
        )
