import copy
import numpy as np


class DiscreteFactor:

    def __init__(self, variables, cardinality, values=None, state_names=None):
        """
        Args
            variables: list,
                variables in the scope of the factor
            cardinality: list,
                cardinalities of each variable, where len(cardinality)=len(variables)
            values: list,
                row vector of values of variables with ordering such that right-most variables
                defined in `variables` cycle through their values the fastest
            state_names: dictionary,
                mapping variables to their states, of format {label_name: ['state1', 'state2']}
        """
        self.variables = list(variables)
        self.cardinality = cardinality
        if values is None:
            self._values = None
        else:
            self._values = np.array(values).reshape(self.cardinality)
        self.state_names = state_names

    def __mul__(self, other):
        return self.product(other)

    @property
    def values(self):
        return self._values

    def update_values(self, new_values):
        """We make this available because _values is allowed to be None on init"""
        self._values = np.array(new_values).reshape(self.cardinality)

    def get_value_for_state_vector(self, dict_of_states):
        """
        Return the value for a dictionary of variable states.

        Args
            dict_of_states: dictionary,
                of format {label_name1: 'state1', label_name2: 'True'}
        Returns
            probability, a float, the factor value for a specific combination of variable states
        """
        assert sorted(dict_of_states.keys()) == sorted(self.variables), \
            "The keys for the dictionary of states must match the variables in factor scope."
        state_coordinates = []
        for var in self.variables:
            var_state = dict_of_states[var]
            idx_in_var_axis = self.state_names[var].index(var_state)
            state_coordinates.append(idx_in_var_axis)
        return self.values[tuple(state_coordinates)]

    def add_new_variables_from_other_factor(self, other):
        """Add new variables to the factor."""
        extra_vars = set(other.variables) - set(self.variables)
        # if all of these variables already exist there is nothing to do
        if len(extra_vars) == 0:
            return
        # otherwise, extend the values array
        slice_ = [slice(None)] * len(self.variables)
        slice_.extend([np.newaxis] * len(extra_vars))
        self._values = self._values[slice_]
        self.variables.extend(extra_vars)

        new_card_var = other.get_cardinality(extra_vars)
        self.cardinality.extend([new_card_var[var] for var in extra_vars])
        return

    def get_cardinality(self, variables):
        return {var: self.cardinality[self.variables.index(var)] for var in variables}

    def product(self, other):
        left = copy.deepcopy(self)

        if isinstance(other, (int, float)):
            # TODO: handle case of multiplication by constant
            pass
        else:
            # assert right is a class or subclass of DiscreteFactor
            # that has attributes: variables, values; method: get_cardinality
            right = copy.deepcopy(other)
            left.add_new_variables_from_other_factor(right)
            right.add_new_variables_from_other_factor(left)

        # reorder variables in right factor to match order in left
        source_axes = list(range(right.values.ndim))
        destination_axes = [right.variables.index(var) for var in left.variables]
        right.variables = [right.variables[idx] for idx in destination_axes]

        # rearrange values in right factor to correspond to the reordered variables
        right._values = np.moveaxis(right.values, source_axes, destination_axes)
        left._values = left.values * right.values
        return left

    def marginalize(self, vars):
        """
        Args
            vars: list,
                variables over which to marginalize the factor
        Returns
            DiscreteFactor
        """
        phi = copy.deepcopy(self)

        var_indexes = []
        for var in vars:
            if var not in phi.variables:
                raise ValueError('{} not in scope'.format(var))
            else:
                var_indexes.append(self.variables.index(var))

        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indexes))
        phi.variables = [self.variables[index] for index in index_to_keep]
        phi.cardinality = [self.cardinality[index] for index in index_to_keep]
        phi._values = np.sum(phi.values, axis=tuple(var_indexes))
        return phi
