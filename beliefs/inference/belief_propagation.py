import numpy as np
from collections import namedtuple

from beliefs.types.Node import InvalidLambdaMsgToParent
from beliefs.utils.math_helper import is_kronecker_delta


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
          model: an instance of BayesianModel class or subclass
          inplace: bool
              modify in-place the nodes in the model during belief propagation
        """
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
        print("Node", node_to_update_label_id)

        node = self.model.nodes[node_to_update_label_id]

        # exclude the message sender (either a parent or child) from getting an
        # outgoing msg from the node to update
        parent_ids = node.parents - set([msg_sender_label_id])
        child_ids = node.children - set([msg_sender_label_id])
        print("parent_ids:", parent_ids)
        print("child_ids:", child_ids)

        if msg_sender_label_id is not None:
            # update triggered by receiving a message, not pinning to evidence
            assert len(node.parents) + len(node.children) - 1 == len(parent_ids) + len(child_ids)

        if node_to_update_label_id not in evidence:
            node.compute_pi_agg()
            print("belief propagation pi_agg", node.pi_agg)
            node.compute_lambda_agg()
            print("belief propagation lambda_agg", node.lambda_agg)

        for parent_id in parent_ids:
            try:
                new_lambda_msg = node.compute_lambda_msg_to_parent(parent_k=parent_id)
            except InvalidLambdaMsgToParent:
                raise ConflictingEvidenceError(evidence=evidence)

            parent_node = self.model.nodes[parent_id]
            parent_node.update_lambda_msg_from_child(child=node_to_update_label_id,
                                                     new_value=new_lambda_msg)
            nodes_to_update.add(MsgPassers(msg_receiver=parent_id,
                                           msg_sender=node_to_update_label_id))

        for child_id in child_ids:
            new_pi_msg = node.compute_pi_msg_to_child(child_k=child_id)
            child_node = self.model.nodes[child_id]
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

        for node in self.model.nodes.values():
            ones_vector = np.ones([node.cardinality])

            node.lambda_agg = ones_vector
            for child in node.lambda_received_msgs.keys():
                node.update_lambda_msg_from_child(child=child,
                                                  new_value=ones_vector)
        print("Finished initializing Lambda(x) and lambda_received_msgs per node.")

        print("Start downward sweep from nodes.  Sending Pi messages only.")
        topdown_order = self.model.get_topologically_sorted_nodes(reverse=False)

        for node_id in topdown_order:
            print('label in iteration through top-down order:', node_id)

            node_sending_msg = self.model.nodes[node_id]
            child_ids = node_sending_msg.children

            if node_sending_msg.pi_agg is None:
                node_sending_msg.compute_pi_agg()

            for child_id in child_ids:
                print("child", child_id)
                new_pi_msg = node_sending_msg.compute_pi_msg_to_child(child_k=child_id)
                print(new_pi_msg)

                child_node = self.model.nodes[child_id]
                child_node.update_pi_msg_from_parent(parent=node_id,
                                                     new_value=new_pi_msg)

    def _run_belief_propagation(self, evidence):
        """
        Input:
          evidence: dict
              a dict key, value pair as {var: state_of_var observed}
        """
        for evidence_id, observed_value in evidence.items():
            nodes_to_update = set()

            if evidence_id not in self.model.nodes.keys():
                raise KeyError("Evidence supplied for non-existent label_id: {}"
                               .format(evidence_id))

            if is_kronecker_delta(observed_value):
                # specific evidence
                self.model.nodes[evidence_id].lambda_agg = observed_value
            else:
                # virtual evidence
                self.model.nodes[evidence_id].lambda_agg = \
                    self.model.nodes[evidence_id].lambda_agg * observed_value

            nodes_to_update.add(MsgPassers(msg_receiver=evidence_id,
                                           msg_sender=None))

            self._belief_propagation(nodes_to_update=nodes_to_update,
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
        >> from label_graph_service.pgm.inference.belief_propagation import BeliefPropagation
        >> from label_graph_service.pgm.models.BernoulliOrModel import BernoulliOrModel
        >> edges = [('1', '3'), ('2', '3'), ('3', '5')]
        >> model = BernoulliOrModel(edges)
        >> infer = BeliefPropagation(model)
        >> result = infer.query({'2': np.array([0, 1])})
        """
        if not self.model.all_nodes_are_fully_initialized:
            self.initialize_model()

        if evidence:
            self._run_belief_propagation(evidence)

        return {label_id: node.belief for label_id, node in self.model.nodes.items()}
