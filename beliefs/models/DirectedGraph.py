import networkx as nx


class DirectedGraph(nx.DiGraph):
    """
    Base class for all directed graphical models.
    """
    def __init__(self, edges=None, node_labels=None):
        """
        Input:
            edges: an edge list, e.g. [(parent1, child1), (parent1, child2)]
            node_labels: a list of strings of node labels
        """
        super().__init__()
        if edges is not None:
            self.add_edges_from(edges)
        if node_labels is not None:
            self.add_nodes_from(node_labels)

    def get_leaves(self):
        """
        Returns a list of leaves of the graph.
        """
        return [node for node, out_degree in self.out_degree() if out_degree == 0]

    def get_roots(self):
        """
        Returns a list of roots of the graph.
        """
        return [node for node, in_degree in self.in_degree() if in_degree == 0]

    def get_topologically_sorted_nodes(self, reverse=False):
        if reverse:
            return list(reversed(list(nx.topological_sort(self))))
        else:
            return nx.topological_sort(self)
