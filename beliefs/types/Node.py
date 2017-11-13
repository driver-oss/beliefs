import numpy as np
from functools import reduce
from enum import Enum


class InvalidLambdaMsgToParent(Exception):
    """Computed invalid lambda msg to send to parent."""
    pass


class MessageType(Enum):
    LAMBDA = 'lambda'
    PI = 'pi'


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
        Input:
            label_id: str
            children: str
            parents: set of strings
            cardinality: int, cardinality of the random variable the node represents
            cpd: an instance of a conditional probability distribution,
                 e.g. BernoulliOrFactor or pgmpy's TabularCPD
        """
        self.label_id = label_id
        self.children = children
        self.parents = parents
        self.cardinality = cardinality
        self.cpd = cpd

        self.pi_agg = None  # np.array dimensions [1, cardinality]
        self.lambda_agg = None  # np.array dimensions [1, cardinality]

        self.pi_received_msgs = self._init_received_msgs(parents)
        self.lambda_received_msgs = self._init_received_msgs(children)

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
        if self.pi_agg.any() and self.lambda_agg.any():
            belief = np.multiply(self.pi_agg, self.lambda_agg)
            return self._normalize(belief)
        else:
            return None

    def _normalize(self, value):
        return value/value.sum()

    @staticmethod
    def _init_received_msgs(keys):
        return {k: None for k in keys}

    def _return_msgs_received_for_msg_type(self, message_type):
        """
        Input:
          message_type: MessageType enum

        Returns:
          msg_values: list of message values (each an np.array)
        """
        if message_type == MessageType.LAMBDA:
            msg_values = [msg for msg in self.lambda_received_msgs.values()]
        elif message_type == MessageType.PI:
            msg_values = [msg for msg in self.pi_received_msgs.values()]
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
                "Missing value for {msg_type} msg from child: can't compute {msg_type}_agg.".
                format(msg_type=message_type.value)
            )
        else:
            return msg_values

    def compute_pi_agg(self):
        # TODO: implement explict factor product operation
        raise NotImplementedError

    def compute_lambda_agg(self):
        if not self.children:
            return self.lambda_agg
        else:
            lambda_msg_values = self.validate_and_return_msgs_received_for_msg_type(MessageType.LAMBDA)
            self.lambda_agg = reduce(np.multiply, lambda_msg_values)
        return self.lambda_agg

    def _update_received_msg_by_key(self, received_msg_dict, key, new_value):
        if key not in received_msg_dict.keys():
            raise ValueError("Label id '{}' to update message isn't in allowed set of keys: {}".
                             format(key, received_msg_dict.keys()))

        if not isinstance(new_value, np.ndarray):
            raise TypeError("Expected a new value of type numpy.ndarray, but got type {}".
                            format(type(new_value)))

        if new_value.shape != (self.cardinality,):
            raise ValueError("Expected new value to be of dimensions ({},) but got {} instead".
                             format(self.cardinality, new_value.shape))
        received_msg_dict[key] = new_value

    def update_pi_msg_from_parent(self, parent, new_value):
        self._update_received_msg_by_key(received_msg_dict=self.pi_received_msgs,
                                         key=parent,
                                         new_value=new_value)

    def update_lambda_msg_from_child(self, child, new_value):
        self._update_received_msg_by_key(received_msg_dict=self.lambda_received_msgs,
                                         key=child,
                                         new_value=new_value)

    def compute_pi_msg_to_child(self, child_k):
        lambda_msg_from_child = self.lambda_received_msgs[child_k]
        if lambda_msg_from_child is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                # 0/0 := 0
                return self._normalize(
                    np.nan_to_num(np.divide(self.belief, lambda_msg_from_child)))
        else:
            raise ValueError("Can't compute pi message to child_{} without having received" \
                             "a lambda message from that child.")

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

        if (self.pi_agg is None) or (self.lambda_agg is None):
            return False

        return True
