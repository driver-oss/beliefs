import copy
import networkx as nx

from beliefs.models.DirectedGraph import DirectedGraph
from beliefs.utils.math_helper import is_kronecker_delta


class BayesianModel(DirectedGraph):
    """
    Bayesian model stores nodes and edges described by conditional probability
    distributions.
    """
    def __init__(self, edges=[], variables=[], cpds=[]):
        """
        Base class for Bayesian model.

        Input:
          edges: (optional) list of edges,
                tuples of form ('parent', 'child')
          variables: (optional) list of str or int
                labels for variables
          cpds: (optional) list of CPDs
                TabularCPD class or subclass
        """
        super().__init__()
        super().add_edges_from(edges)
        super().add_nodes_from(variables)
        self.cpds = cpds

    def copy(self):
        """
        Returns a copy of the model.
        """
        copy_model = self.__class__(edges=list(self.edges()).copy(),
                                    variables=list(self.nodes()).copy(),
                                    cpds=[cpd.copy() for cpd in self.cpds])
        return copy_model

    def get_variables_in_definite_state(self):
        """
        Returns a set of labels of all nodes in a definite state, i.e. with
        label values that are kronecker deltas.

        RETURNS
          set of strings (labels)
        """
        return {label for label, node in self.nodes_dict.items() if is_kronecker_delta(node.belief)}

    def get_unobserved_variables_in_definite_state(self, observed=set()):
        """
        Returns a set of labels that are inferred to be in definite state, given
        list of labels that were directly observed (e.g. YES/NOs, but not MAYBEs).

        INPUT
          observed: set of strings, directly observed labels
        RETURNS
          set of strings, labels inferred to be in a definite state
        """

        # Assert that beliefs of directly observed vars are kronecker deltas
        for label in observed:
            assert is_kronecker_delta(self.nodes_dict[label].belief), \
                ("Observed label has belief {} but should be kronecker delta"
                 .format(self.nodes_dict[label].belief))

        vars_in_definite_state = self.get_variables_in_definite_state()
        assert observed <= vars_in_definite_state, \
            "Expected set of observed labels to be a subset of labels in definite state."
        return vars_in_definite_state - observed

    def _get_ancestors_of(self, observed):
        """Return list of ancestors of observed labels, including the observed labels themselves."""
        ancestors = observed.copy()
        for label in observed:
            ancestors.update(nx.ancestors(self, label))
        return ancestors

    def reachable_observed_variables(self, source, observed=set()):
        """
        Returns list of observed labels (labels with direct evidence to be in a definite
        state) that are reachable from the source.

        INPUT
          source: string, label of node for which to evaluate reachable observed labels
          observed: set of strings, directly observed labels
        RETURNS
          reachable_observed_vars: set of strings, observed labels (variables with direct
              evidence) that are reachable from the source label.
        """
        ancestors_of_observed = self._get_ancestors_of(observed)

        visit_list = set()
        visit_list.add((source, 'up'))
        traversed_list = set()
        reachable_observed_vars = set()

        while visit_list:
            node, direction = visit_list.pop()
            if (node, direction) not in traversed_list:
                if node in observed:
                    reachable_observed_vars.add(node)
                traversed_list.add((node, direction))
                if direction == 'up' and node not in observed:
                    for parent in self.predecessors(node):
                        # causal flow
                        visit_list.add((parent, 'up'))
                    for child in self.successors(node):
                        # common cause flow
                        visit_list.add((child, 'down'))
                elif direction == 'down':
                    if node not in observed:
                        # evidential flow
                        for child in self.successors(node):
                            visit_list.add((child, 'down'))
                    if node in ancestors_of_observed:
                        # common effect flow (activated v-structure)
                        for parent in self.predecessors(node):
                            visit_list.add((parent, 'up'))
        return reachable_observed_vars
