import copy
from enum import Enum
import numpy as np
import itertools
from functools import reduce

import networkx as nx

from beliefs.models.base_models import BayesianModel
from beliefs.factors.discrete_factor import DiscreteFactor
from beliefs.factors.bernoulli_or_cpd import BernoulliOrCPD
from beliefs.factors.bernoulli_and_cpd import BernoulliAndCPD


class InvalidLambdaMsgToParent(Exception):
    """Computed invalid lambda msg to send to parent."""
    pass


class MessageType(Enum):
    LAMBDA = 'lambda'
    PI = 'pi'


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
    def init_from_edges(cls, edges, node_class):
        """Create nodes from the same node class.

        Input:
           edges: list of edge tuples of form ('parent', 'child')
           node_class: the Node class or subclass from which to
                 create all the nodes from edges.
        """
        nodes = set()
        g = nx.DiGraph(edges)

        for label in set(itertools.chain(*edges)):
            node = node_class(label_id=label,
                              children=list(g.successors(label)),
                              parents=list(g.predecessors(label)))
            nodes.add(node)
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
            self.nodes_dict[root].update_pi_agg(self.nodes_dict[root].cpd.values)

        for leaf in self.get_leaves():
            self.nodes_dict[leaf].update_lambda_agg(np.ones([self.nodes_dict[leaf].cardinality]))

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


class Node:
    """A node in a DAG with methods to compute the belief (marginal probability
    of the node given evidence) and compute pi/lambda messages to/from its neighbors
    to update its belief.

    Implemented from Pearl's belief propagation algorithm.
    """
    def __init__(self,
                 label_id,
                 children,
                 parents,
                 cardinality,
                 cpd):
        """
        Args
            label_id: str
            children: set of strings
            parents: set of strings
            cardinality: int, cardinality of the random variable the node represents
            cpd: an instance of a conditional probability distribution,
                 e.g. BernoulliOrCPD or TabularCPD
        """
        self.label_id = label_id   # this can be obtained from cpd.variable
        self.children = children
        self.parents = parents   # this can be obtained from cpd.variables[1:]
        self.cardinality = cardinality  # this can be obtained from cpd.cardinality[0]
        self.cpd = cpd

        # instances of DiscreteFactor with `values` an np.array of dimensions [1, cardinality]
        self.pi_agg = self._init_aggregate_values()
        self.lambda_agg = self._init_aggregate_values()

        self.pi_received_msgs = self._init_pi_received_msgs(parents)
        self.lambda_received_msgs = {child: self._init_aggregate_values() for child in children}

    @classmethod
    def from_cpd_class(cls,
                       label_id,
                       children,
                       parents,
                       cardinality,
                       cpd_class):
        cpd = cpd_class(label_id, parents)
        return cls(label_id, children, parents, cardinality, cpd)

    @property
    def belief(self):
        if any(self.pi_agg.values) and any(self.lambda_agg.values):
            belief = (self.lambda_agg * self.pi_agg).values
            return self._normalize(belief)
        else:
            return None

    def _normalize(self, value):
        return value/value.sum()

    def _init_aggregate_values(self):
        return DiscreteFactor(variables=[self.cpd.variable],
                              cardinality=[self.cardinality],
                              values=None,
                              state_names=None)

    def _init_pi_received_msgs(self, parents):
        msgs = {}
        for k in parents:
            kth_cardinality = self.cpd.cardinality[self.cpd.variables.index(k)]
            msgs[k] = DiscreteFactor(variables=[k],
                                     cardinality=[kth_cardinality],
                                     values=None,
                                     state_names=None)
        return msgs

    def _return_msgs_received_for_msg_type(self, message_type):
        """
        Input:
          message_type: MessageType enum

        Returns:
          msg_values: list of message values (each an np.array)
        """
        if message_type == MessageType.LAMBDA:
            msg_values = [msg.values for msg in self.lambda_received_msgs.values()]
        elif message_type == MessageType.PI:
            msg_values = [msg.values for msg in self.pi_received_msgs.values()]
        return msg_values

    def validate_and_return_msgs_received_for_msg_type(self, message_type):
        """
        Check that all messages have been received from children (parents).
        Raise error if all messages have not been received.  Called
        before calculating lambda_agg (pi_agg).

        Input:
          message_type: MessageType enum

        Returns:
          msg_values: list of message values (each an np.array)
        """
        msg_values = self._return_msgs_received_for_msg_type(message_type)

        if any(msg is None for msg in msg_values):
            raise ValueError(
                "Missing value for {msg_type} msg from child: can't compute {msg_type}_agg."
                .format(msg_type=message_type.value)
            )
        else:
            return msg_values

    def compute_pi_agg(self):
        # TODO: implement explict factor product operation
        raise NotImplementedError

    def compute_lambda_agg(self):
        if len(self.children) == 0:
            return self.lambda_agg.values
        else:
            lambda_msg_values =\
                    self.validate_and_return_msgs_received_for_msg_type(MessageType.LAMBDA)
            self.update_lambda_agg(reduce(np.multiply, lambda_msg_values))
        return self.lambda_agg.values

    def update_pi_agg(self, new_value):
        self.pi_agg.update_values(np.array(new_value).reshape(self.cardinality))

    def update_lambda_agg(self, new_value):
        self.lambda_agg.update_values(np.array(new_value).reshape(self.cardinality))

    def _update_received_msg_by_key(self, received_msg_dict, key, new_value, message_type):
        if key not in received_msg_dict.keys():
            raise ValueError("Label id '{}' to update message isn't in allowed set of keys: {}"
                             .format(key, received_msg_dict.keys()))

        if not isinstance(new_value, np.ndarray):
            raise TypeError("Expected a new value of type numpy.ndarray, but got type {}"
                            .format(type(new_value)))

        if message_type == MessageType.LAMBDA:
            expected_shape = (self.cardinality,)
        elif message_type == MessageType.PI:
            expected_shape = (self.cpd.cardinality[self.cpd.variables.index(key)],)

        if new_value.shape != expected_shape:
                raise ValueError("Expected new value to be of dimensions ({},) but got {} instead"
                                 .format(expected_shape, new_value.shape))
        received_msg_dict[key]._values = new_value

    def update_pi_msg_from_parent(self, parent, new_value):
        self._update_received_msg_by_key(received_msg_dict=self.pi_received_msgs,
                                         key=parent,
                                         new_value=new_value,
                                         message_type=MessageType.PI)

    def update_lambda_msg_from_child(self, child, new_value):
        self._update_received_msg_by_key(received_msg_dict=self.lambda_received_msgs,
                                         key=child,
                                         new_value=new_value,
                                         message_type=MessageType.LAMBDA)

    def compute_pi_msg_to_child(self, child_k):
        lambda_msg_from_child = self.lambda_received_msgs[child_k].values
        if lambda_msg_from_child is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                # 0/0 := 0
                return self._normalize(
                    np.nan_to_num(np.divide(self.belief, lambda_msg_from_child)))
        else:
            raise ValueError("Can't compute pi message to child_{} without having received a lambda message from that child.")

    def compute_lambda_msg_to_parent(self, parent_k):
        # TODO: implement explict factor product operation
        raise NotImplementedError

    @property
    def is_fully_initialized(self):
        """
        Returns True if all lambda and pi messages and lambda_agg and
        pi_agg are not None, else False.
        """
        lambda_msgs = self._return_msgs_received_for_msg_type(MessageType.LAMBDA)
        if any(msg is None for msg in lambda_msgs):
            return False

        pi_msgs = self._return_msgs_received_for_msg_type(MessageType.PI)
        if any(msg is None for msg in pi_msgs):
            return False

        if (self.pi_agg.values is None) or (self.lambda_agg.values is None):
            return False

        return True


class BernoulliOrNode(Node):
    def __init__(self,
                 label_id,
                 children,
                 parents):
        super().__init__(label_id=label_id,
                         children=children,
                         parents=parents,
                         cardinality=2,
                         cpd=BernoulliOrCPD(label_id, parents))

    def compute_pi_agg(self):
        if len(self.parents) == 0:
            self.update_pi_agg(self.cpd.values)
        else:
            pi_msg_values = self.validate_and_return_msgs_received_for_msg_type(MessageType.PI)
            parents_p0 = [p[0] for p in pi_msg_values]
            # Parents are Bernoulli variables with pi_msg_values (surrogate prior probabilities)
            # of p = [P(False), P(True)]
            p_0 = reduce(lambda x, y: x*y, parents_p0)
            p_1 = 1 - p_0
            self.update_pi_agg(np.array([p_0, p_1]))
        return self.pi_agg

    def compute_lambda_msg_to_parent(self, parent_k):
        if np.array_equal(self.lambda_agg.values, np.ones([self.cardinality])):
            return np.ones([self.cardinality])
        else:
            # TODO: cleanup this validation
            _ = self.validate_and_return_msgs_received_for_msg_type(MessageType.PI)
            p0_excluding_k = [p.values[0] for par_id, p in self.pi_received_msgs.items() if par_id != parent_k]
            p0_product = reduce(lambda x, y: x*y, p0_excluding_k, 1)
            lambda_0 = self.lambda_agg.values[1] + (self.lambda_agg.values[0] - self.lambda_agg.values[1])*p0_product
            lambda_1 = self.lambda_agg.values[1]
            lambda_msg = np.array([lambda_0, lambda_1])
            if not any(lambda_msg):
                raise InvalidLambdaMsgToParent
            return self._normalize(lambda_msg)


class BernoulliAndNode(Node):
    def __init__(self,
                 label_id,
                 children,
                 parents):
        super().__init__(label_id=label_id,
                         children=children,
                         parents=parents,
                         cardinality=2,
                         cpd=BernoulliAndCPD(label_id, parents))

    def compute_pi_agg(self):
        if len(self.parents) == 0:
            self.update_pi_agg(self.cpd.values)
        else:
            pi_msg_values = self.validate_and_return_msgs_received_for_msg_type(MessageType.PI)
            parents_p1 = [p[1] for p in pi_msg_values]
            # Parents are Bernoulli variables with pi_msg_values (surrogate prior probabilities)
            # of p = [P(False), P(True)]
            p_1 = reduce(lambda x, y: x*y, parents_p1)
            p_0 = 1 - p_1
            self.update_pi_agg(np.array([p_0, p_1]))
        return self.pi_agg

    def compute_lambda_msg_to_parent(self, parent_k):
        if np.array_equal(self.lambda_agg.values, np.ones([self.cardinality])):
            return np.ones([self.cardinality])
        else:
            # TODO: cleanup this validation
            _ = self.validate_and_return_msgs_received_for_msg_type(MessageType.PI)
            p1_excluding_k = [p.values[1] for par_id, p in self.pi_received_msgs.items() if par_id != parent_k]
            p1_product = reduce(lambda x, y: x*y, p1_excluding_k, 1)
            lambda_0 = self.lambda_agg.values[0]
            lambda_1 = self.lambda_agg.values[0] + (self.lambda_agg.values[1] - self.lambda_agg.values[0])*p1_product
            lambda_msg = np.array([lambda_0, lambda_1])
            if not any(lambda_msg):
                raise InvalidLambdaMsgToParent
            return self._normalize(lambda_msg)
