import numpy as np
from collections import namedtuple
import logging

from beliefs.models.belief_update_node_model import (
    InvalidLambdaMsgToParent,
    BeliefUpdateNodeModel
)
from beliefs.utils.math_helper import is_kronecker_delta


logger = logging.getLogger(__name__)


MsgPassers = namedtuple('MsgPassers', ['msg_receiver', 'msg_sender'])


class ConflictingEvidenceError(Exception):
    """Failed to run belief propagation on label graph because of conflicting evidence."""
    def __init__(self, evidence):
        message = (
            "Can't run belief propagation with conflicting evidence: {}"
            .format(evidence)
        )
        super().__init__(message)


class BeliefPropagation:
    def __init__(self, model, inplace=True):
        """
        Input:
          model: an instance of BeliefUpdateNodeModel
          inplace: bool
              modify in-place the nodes in the model during belief propagation
        """
        if not isinstance(model, BeliefUpdateNodeModel):
            raise TypeError("Model must be an instance of BeliefUpdateNodeModel")
        if inplace is False:
            self.model = model.copy()
        else:
            self.model = model

    def _belief_propagation(self, nodes_to_update, evidence):
        """
        Implementation of Pearl's belief propagation algorithm for polytrees.

        ref: "Fusion, Propagation, and Structuring in Belief Networks"
             Artificial Intelligence 29 (1986) 241-288

        Input:
          nodes_to_update: list
               list of MsgPasser namedtuples.
          evidence: dict,
               a dict key, value pair as {var: state_of_var observed}
        """
        if len(nodes_to_update) == 0:
            return

        node_to_update_label_id, msg_sender_label_id = nodes_to_update.pop()
        logging.info("Node: %s", node_to_update_label_id)

        node = self.model.nodes_dict[node_to_update_label_id]

        # exclude the message sender (either a parent or child) from getting an
        # outgoing msg from the node to update
        parent_ids = set(node.parents) - set([msg_sender_label_id])
        child_ids = set(node.children) - set([msg_sender_label_id])
        logging.info("parent_ids: %s", str(parent_ids))
        logging.info("child_ids: %s", str(child_ids))

        if msg_sender_label_id is not None:
            # update triggered by receiving a message, not pinning to evidence
            assert len(node.parents) + len(node.children) - 1 == len(parent_ids) + len(child_ids)

        if node_to_update_label_id not in evidence:
            node.compute_pi_agg()
            logging.info("belief propagation pi_agg: %s", np.array2string(node.pi_agg.values))
            node.compute_lambda_agg()
            logging.info("belief propagation lambda_agg: %s", np.array2string(node.lambda_agg.values))

        for parent_id in parent_ids:
            try:
                new_lambda_msg = node.compute_lambda_msg_to_parent(parent_k=parent_id)
            except InvalidLambdaMsgToParent:
                raise ConflictingEvidenceError(evidence=evidence)

            parent_node = self.model.nodes_dict[parent_id]
            parent_node.update_lambda_msg_from_child(child=node_to_update_label_id,
                                                     new_value=new_lambda_msg)
            nodes_to_update.add(MsgPassers(msg_receiver=parent_id,
                                           msg_sender=node_to_update_label_id))

        for child_id in child_ids:
            new_pi_msg = node.compute_pi_msg_to_child(child_k=child_id)
            child_node = self.model.nodes_dict[child_id]
            child_node.update_pi_msg_from_parent(parent=node_to_update_label_id,
                                                 new_value=new_pi_msg)
            nodes_to_update.add(MsgPassers(msg_receiver=child_id,
                                           msg_sender=node_to_update_label_id))
        self._belief_propagation(nodes_to_update, evidence)

    def initialize_model(self):
        """
        Apply boundary conditions:
            - Set pi_agg equal to prior probabilities for root nodes.
            - Set lambda_agg equal to vector of ones for leaf nodes.

        - Set lambda_agg, lambda_received_msgs to vectors of ones (same effect as
          actually passing lambda messages up from leaf nodes to root nodes).
        - Calculate pi_agg and pi_received_msgs for all nodes without evidence.
          (Without evidence, belief equals pi_agg.)
        """
        self.model.set_boundary_conditions()

        for node in self.model.nodes_dict.values():
            ones_vector = np.ones([node.cardinality])
            node.update_lambda_agg(ones_vector)

            for child in node.lambda_received_msgs.keys():
                node.update_lambda_msg_from_child(child=child,
                                                  new_value=ones_vector)
        logging.info("Finished initializing Lambda(x) and lambda_received_msgs per node.")

        logging.info("Start downward sweep from nodes.  Sending Pi messages only.")
        topdown_order = self.model.get_topologically_sorted_nodes(reverse=False)

        for node_id in topdown_order:
            logging.info('label in iteration through top-down order: %s', str(node_id))

            node_sending_msg = self.model.nodes_dict[node_id]
            child_ids = node_sending_msg.children

            if node_sending_msg.pi_agg.values is None:
                node_sending_msg.compute_pi_agg()

            for child_id in child_ids:
                logging.info("child: %s", str(child_id))
                new_pi_msg = node_sending_msg.compute_pi_msg_to_child(child_k=child_id)
                logging.info("new_pi_msg: %s", np.array2string(new_pi_msg))

                child_node = self.model.nodes_dict[child_id]
                child_node.update_pi_msg_from_parent(parent=node_id,
                                                     new_value=new_pi_msg)

    def _run_belief_propagation(self, evidence):
        """
        Input:
          evidence: dict
              a dict key, value pair as {var: state_of_var observed}
        """
        for evidence_id, observed_value in evidence.items():
            if evidence_id not in self.model.nodes_dict.keys():
                raise KeyError("Evidence supplied for non-existent label_id: {}"
                               .format(evidence_id))

            if is_kronecker_delta(observed_value):
                # specific evidence
                self.model.nodes_dict[evidence_id].update_lambda_agg(observed_value)
            else:
                # virtual evidence
                self.model.nodes_dict[evidence_id].update_lambda_agg(
                    self.model.nodes_dict[evidence_id].lambda_agg.values * observed_value
                )
            nodes_to_update = [MsgPassers(msg_receiver=evidence_id, msg_sender=None)]
            self._belief_propagation(nodes_to_update=set(nodes_to_update),
                                     evidence=evidence)

    def query(self, evidence={}):
        """
        Run belief propagation given evidence.

        Input:
          evidence: dict
              a dict key, value pair as {var: state_of_var observed},
              e.g. {'3': np.array([0,1])} if label '3' is True.

        Returns:
          beliefs: dict
              a dict key, value pair as {var: belief}

        Example
        -------
        >> import numpy as np
        >> from beliefs.inference.belief_propagation import BeliefPropagation
        >> from beliefs.models.belief_update_node_model import BeliefUpdateNodeModel, BernoulliOrNode
        >> edges = [('1', '3'), ('2', '3'), ('3', '5')]
        >> model = BeliefUpdateNodeModel.init_from_edges(edges, BernoulliOrNode)
        >> infer = BeliefPropagation(model)
        >> result = infer.query(evidence={'2': np.array([0, 1])})
        """
        if not self.model.all_nodes_are_fully_initialized:
            self.initialize_model()

        if evidence:
            self._run_belief_propagation(evidence)

        return {label_id: node.belief for label_id, node in self.model.nodes_dict.items()}
