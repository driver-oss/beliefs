"""
The MIT License (MIT)

Copyright (c) 2013-2017 pgmpy

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import networkx as nx

from beliefs.utils.math_helper import is_kronecker_delta


class DirectedGraph(nx.DiGraph):
    """
    Base class for all directed graphical models.
    """
    def __init__(self, edges=None, node_labels=None):
        """
        Args
            edges: list,
               a list of edge tuples, e.g. [(parent1, child1), (parent1, child2)]
            node_labels: list,
               a list of strings or integers representing node label ids
        """
        super().__init__()
        if edges is not None:
            self.add_edges_from(edges)
        if node_labels is not None:
            self.add_nodes_from(node_labels)

    def get_leaves(self):
        """Return a list of leaves of the graph"""
        return [node for node, out_degree in self.out_degree() if out_degree == 0]

    def get_roots(self):
        """Return a list of roots of the graph"""
        return [node for node, in_degree in self.in_degree() if in_degree == 0]

    def get_topologically_sorted_nodes(self, reverse=False):
        """Return a list of nodes in topological sort order"""
        if reverse:
            return list(reversed(list(nx.topological_sort(self))))
        else:
            return nx.topological_sort(self)


class BayesianModel(DirectedGraph):
    """
    Bayesian model stores nodes and edges described by conditional probability
    distributions.
    """
    def __init__(self, edges=[], variables=[], cpds=[]):
        """
        Base class for Bayesian model.

        Args
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
        """Return a copy of the model"""
        return self.__class__(edges=list(self.edges()).copy(),
                              variables=list(self.nodes()).copy(),
                              cpds=[cpd.copy() for cpd in self.cpds])

    def get_variables_in_definite_state(self):
        """
        Get labels of all nodes in a definite state, i.e. with label values
        that are kronecker deltas.

        Returns
          set of strings (labels)
        """
        return {label for label, node in self.nodes_dict.items() if is_kronecker_delta(node.belief)}

    def get_unobserved_variables_in_definite_state(self, observed=set()):
        """
        Returns a set of labels that are inferred to be in definite state, given
        list of labels that were directly observed (e.g. YES/NOs, but not MAYBEs).

        Args
            observed: set,
                set of strings, directly observed labels
        Returns
            set of strings, the labels inferred to be in a definite state
        """
        for label in observed:
            # beliefs of directly observed vars should be kronecker deltas
            assert is_kronecker_delta(self.nodes_dict[label].belief), \
                ("Observed label has belief {} but should be kronecker delta"
                 .format(self.nodes_dict[label].belief))

        vars_in_definite_state = self.get_variables_in_definite_state()
        assert observed <= vars_in_definite_state, \
            "Expected set of observed labels to be a subset of labels in definite state."
        return vars_in_definite_state - observed

    def _get_ancestors_of(self, labels):
        """
        Get set of ancestors of an iterable of labels.

        Args
            observed: iterable,
                label ids for which ancestors should be retrieved

        Returns
            ancestors: set,
                set of label ids of ancestors of the input labels
        """
        ancestors = set()
        for label in labels:
            ancestors.update(nx.ancestors(self, label))
        return ancestors

    def reachable_observed_variables(self, source, observed=set()):
        """
        Get list of directly observed labels (labels with evidence in a definite
        state) that are reachable from the source.

        Args
            source: string,
                label of node for which to evaluate reachable observed labels
            observed: set,
                set of strings, directly observed labels
        Returns
            reachable_observed_vars: set,
                set of strings, observed labels (variables with direct evidence)
                that are reachable from the source label
        """
        ancestors_of_observed = self._get_ancestors_of(observed)
        ancestors_of_observed.update(observed)  # include observed labels

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
