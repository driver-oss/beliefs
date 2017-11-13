import copy
import numpy as np
import networkx as nx

from beliefs.models.DirectedGraph import DirectedGraph
from beliefs.utils.edges_helper import EdgesHelper
from beliefs.utils.math_helper import is_kronecker_delta


class BayesianModel(DirectedGraph):
    """
    Bayesian model stores nodes and edges described by conditional probability
    distributions.
    """
    def __init__(self, edges, nodes=None):
        """
        Input:
          edges: list of edge tuples of form ('parent', 'child')
          nodes: (optional) dict
            a dict key, value pair as {label_id: instance_of_node_class_or_subclass}
        """
        if nodes is not None:
            super().__init__(edges, nodes.keys())
        else:
            super().__init__(edges)
        self.nodes = nodes

    @classmethod
    def from_node_class(cls, edges, node_class):
        """Automatically create all nodes from the same node class

        Input:
           edges: list of edge tuples of form ('parent', 'child')
           node_class: (optional) the Node class or subclass from which to
                 create all the nodes from edges.
        """
        nodes = cls.create_nodes(edges, node_class)
        return cls.__init__(edges=edges, nodes=nodes)

    @staticmethod
    def create_nodes(edges, node_class):
        """Returns list of Node instances created from edges using
        the default node_class"""
        edges_helper = EdgesHelper(edges)
        nodes = edges_helper.create_nodes_from_edges(node_class=node_class)
        label_to_node = dict()
        for node in nodes:
            label_to_node[node.label_id] = node
        return label_to_node

    def set_boundary_conditions(self):
        """
        1. Root nodes: if x is a node with no parents, set Pi(x) = prior
        probability of x.

        2. Leaf nodes: if x is a node with no children, set Lambda(x)
        to an (unnormalized) unit vector, of length the cardinality of x.
        """
        for root in self.get_roots():
            self.nodes[root].pi_agg = self.nodes[root].cpd.values

        for leaf in self.get_leaves():
            self.nodes[leaf].lambda_agg = np.ones([self.nodes[leaf].cardinality])

    @property
    def all_nodes_are_fully_initialized(self):
        """
        Returns True if, for all nodes in the model, all lambda and pi
        messages and lambda_agg and pi_agg are not None, else False.
        """
        for node in self.nodes.values():
            if not node.is_fully_initialized:
                return False
        return True

    def copy(self):
        """
        Returns a copy of the model.
        """
        copy_edges = self.edges().copy()
        copy_nodes = copy.deepcopy(self.nodes)
        copy_model = self.__class__(edges=copy_edges, nodes=copy_nodes)
        return copy_model

    def get_variables_in_definite_state(self):
        """
        Returns a set of labels of all nodes in a definite state, i.e. with
        label values that are kronecker deltas.

        RETURNS
          set of strings (labels)
        """
        return {label for label, node in self.nodes.items() if is_kronecker_delta(node.belief)}

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
            assert is_kronecker_delta(self.nodes[label].belief), \
                ("Observed label has belief {} but should be kronecker delta"
                 .format(self.nodes[label].belief))

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
