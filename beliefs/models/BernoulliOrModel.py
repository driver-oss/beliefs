from beliefs.models.BayesianModel import BayesianModel
from beliefs.types.BernoulliOrNode import BernoulliOrNode


class BernoulliOrModel(BayesianModel):
    """
    BernoulliOrModel stores node instances of BernoulliOrNodes (Bernoulli
    variables associated with an OR conditional probability distribution).
    """
    def __init__(self, edges, nodes=None):
        """
        Input:
          edges: an edge list, e.g. [(parent1, child1), (parent1, child2)]
        """
        if nodes is None:
            nodes = self.create_nodes(edges, node_class=BernoulliOrNode)
        super().__init__(edges, nodes=nodes)
