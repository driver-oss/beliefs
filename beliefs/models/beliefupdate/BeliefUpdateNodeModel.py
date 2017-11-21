import copy
import numpy as np

from beliefs.models.BayesianModel import BayesianModel
from beliefs.utils.edges_helper import EdgesHelper


class BeliefUpdateNodeModel(BayesianModel):
    """
    A Bayesian model storing nodes (e.g. Node or BernoulliOrNode) implementing properties
    and methods for Pearl's belief update algorithm.

    ref: "Fusion, Propagation, and Structuring in Belief Networks"
          Artificial Intelligence 29 (1986) 241-288

    """
    def __init__(self, nodes_dict):
        """
        Input:
          nodes_dict: dict
            a dict key, value pair as {label_id: instance_of_node_class_or_subclass}
        """
        super().__init__(edges=self._get_edges_from_nodes(nodes_dict.values()),
                         variables=list(nodes_dict.keys()),
                         cpds=[node.cpd for node in nodes_dict.values()])

        self.nodes_dict = nodes_dict

    @classmethod
    def from_edges(cls, edges, node_class):
        """Create nodes from the same node class.

        Input:
           edges: list of edge tuples of form ('parent', 'child')
           node_class: the Node class or subclass from which to
                 create all the nodes from edges.
        """
        edges_helper = EdgesHelper(edges)
        nodes = edges_helper.create_nodes_from_edges(node_class)
        nodes_dict = {node.label_id: node for node in nodes}
        return cls(nodes_dict)

    @staticmethod
    def _get_edges_from_nodes(nodes):
        """
        Return list of all directed edges in nodes.

        Args:
            nodes: an iterable of objects of the Node class or subclass
        Returns:
            edges: list of edge tuples
        """
        edges = set()
        for node in nodes:
            if node.parents:
                edge_tuples = zip(node.parents, [node.label_id]*len(node.parents))
                edges.update(edge_tuples)
        return list(edges)

    def set_boundary_conditions(self):
        """
        1. Root nodes: if x is a node with no parents, set Pi(x) = prior
        probability of x.

        2. Leaf nodes: if x is a node with no children, set Lambda(x)
        to an (unnormalized) unit vector, of length the cardinality of x.
        """
        for root in self.get_roots():
            self.nodes_dict[root].pi_agg = self.nodes_dict[root].cpd.values

        for leaf in self.get_leaves():
            self.nodes_dict[leaf].lambda_agg = np.ones([self.nodes_dict[leaf].cardinality])

    @property
    def all_nodes_are_fully_initialized(self):
        """
        Returns True if, for all nodes in the model, all lambda and pi
        messages and lambda_agg and pi_agg are not None, else False.
        """
        for node in self.nodes_dict.values():
            if not node.is_fully_initialized:
                return False
        return True

    def copy(self):
        """
        Returns a copy of the model.
        """
        copy_nodes = copy.deepcopy(self.nodes_dict)
        copy_model = self.__class__(nodes_dict=copy_nodes)
        return copy_model
