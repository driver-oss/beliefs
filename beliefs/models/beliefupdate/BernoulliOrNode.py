import numpy as np
from functools import reduce

from beliefs.models.beliefupdate.Node import (
    Node,
    MessageType,
    InvalidLambdaMsgToParent
)
from beliefs.factors.BernoulliOrCPD import BernoulliOrCPD


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
        if not self.parents:
            self.pi_agg = self.cpd.values
        else:
            pi_msg_values = self.validate_and_return_msgs_received_for_msg_type(MessageType.PI)
            parents_p0 = [p[0] for p in pi_msg_values]
            p_0 = reduce(lambda x, y: x*y, parents_p0)
            p_1 = 1 - p_0
            self.pi_agg = np.array([p_0, p_1])
        return self.pi_agg

    def compute_lambda_msg_to_parent(self, parent_k):
        if np.array_equal(self.lambda_agg, np.ones([self.cardinality])):
            return np.ones([self.cardinality])
        else:
            # TODO: cleanup this validation
            _ = self.validate_and_return_msgs_received_for_msg_type(MessageType.PI)
            p0_excluding_k = [msg[0] for par_id, msg in self.pi_received_msgs.items() if par_id != parent_k]
            p0_product = reduce(lambda x, y: x*y, p0_excluding_k, 1)
            lambda_0 = self.lambda_agg[1] + (self.lambda_agg[0] - self.lambda_agg[1])*p0_product
            lambda_1 = self.lambda_agg[1]
            lambda_msg = np.array([lambda_0, lambda_1])
            if not any(lambda_msg):
                raise InvalidLambdaMsgToParent
            return self._normalize(lambda_msg)
