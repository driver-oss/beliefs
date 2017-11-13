from collections import defaultdict

from beliefs.types.Node import Node
from beliefs.factors.BernoulliOrFactor import BernoulliOrFactor


class EdgesHelper:
    """Class with convenience methods for working with edges."""
    def __init__(self, edges):
        self.edges = edges

    def get_label_to_children_dict(self):
        """returns dictionary keyed on label, with value a set of children"""
        label_to_children_dict = defaultdict(set)
        for parent, child in self.edges:
            label_to_children_dict[parent].add(child)
        return label_to_children_dict

    def get_label_to_parents_dict(self):
        """returns dictionary keyed on label, with value a set of parents
        Only used to help create dummy factors from edges (not for algo).
        """
        label_to_parents_dict = defaultdict(set)

        for parent, child in self.edges:
            label_to_parents_dict[child].add(parent)
        return label_to_parents_dict

    def get_labels_from_edges(self):
        """Return the set of labels that comprise the vertices of a list of edge tuples."""
        all_labels = set()
        for parent, child in self.edges:
            all_labels.update({parent, child})
        return all_labels

    def create_cpds_from_edges(self, CPD=BernoulliOrFactor):
        """
        Create factors from list of edges.

        Input:
          cpd: a factor class, assumed initialization takes in a label_id, the label_id of
               the child (should = label_id of the node), and set of label_ids of parents.

        Returns:
          factors: a set of (unique) factors of the graph
        """
        labels = self.get_labels_from_edges()
        label_to_parents = self.get_label_to_parents_dict()

        factors = set()

        for label in labels:
            parents = label_to_parents[label]
            cpd = CPD(label, parents)
            factors.add(cpd)
        return factors

    def get_label_to_factor_dict(self, CPD=BernoulliOrFactor):
        """Create a dictionary mapping each label_id to the CPD/factor where
        that label_id is a child.

        Returns:
          label_to_factor: dict mapping each label to the cpd that
                          has that label as a child.
        """
        factors = self.create_cpds_from_edges(CPD=CPD)

        label_to_factor = dict()
        for factor in factors:
            label_to_factor[factor.child] = factor
        return label_to_factor

    def get_label_to_node_dict(self, CPD=BernoulliOrFactor):
        """Create a dictionary mapping each label_id to a Node instance.

        Returns:
          label_to_node: dict mapping each label to the node that has that
                         label as a label_id.
        """
        nodes = self.create_nodes_from_edges()

        label_to_node = dict()
        for node in nodes:
            label_to_node[node.label_id] = node
        return label_to_node

    def get_label_to_node_dict_for_manual_cpds(self, cpds_list):
        """Create a dictionary mapping each label_id to a node that is
        instantiated with a manually defined pgmpy factor instance.

        Input:
          cpds_list - list of instances of pgmpy factors, e.g. TabularCPD

        Returns:
          label_to_node: dict mapping each label to the node that has that
                         label as a label_id.
        """
        label_to_children = self.get_label_to_children_dict()
        label_to_parents = self.get_label_to_parents_dict()

        label_to_node = dict()
        for cpd in cpds_list:
            label_id = cpd.variable

            node = Node(label_id=label_id,
                        children=label_to_children[label_id],
                        parents=label_to_parents[label_id],
                        cardinality=2,
                        cpd=cpd)
            label_to_node[label_id] = node

        return label_to_node

    def create_nodes_from_edges(self, node_class):
        """
        Create instances of the node_class.  Assumes the node class is
        initialized by label_id, children, and parents.

        Returns:
          nodes: a set of (unique) nodes of the graph
        """
        labels = self.get_labels_from_edges()
        labels_to_parents = self.get_label_to_parents_dict()
        labels_to_children = self.get_label_to_children_dict()

        nodes = set()

        for label in labels:
            parents = labels_to_parents[label]
            children = labels_to_children[label]

            node = node_class(label_id=label,
                              children=children,
                              parents=parents)
            nodes.add(node)
        return nodes
