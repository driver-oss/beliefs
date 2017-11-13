import numpy as np
import pytest
from pytest import approx

from beliefs.inference.belief_propagation import BeliefPropagation, ConflictingEvidenceError
from beliefs.models.BernoulliOrModel import BernoulliOrModel
from beliefs.types.BernoulliOrNode import BernoulliOrNode


@pytest.fixture(scope='module')
def edges_four_nodes():
    """Edges define a polytree with 4 nodes (connected in an X-shape with the
    node, 'x', at the center of the X."""
    edges = [('u', 'x'), ('v', 'x'), ('x', 'y'), ('x', 'z')]
    return edges


@pytest.fixture(scope='module')
def simple_edges():
    """Edges define a polytree with 15 nodes."""
    edges = [('1', '3'), ('2', '3'), ('3', '5'), ('4', '5'), ('5', '10'),
             ('5', '9'), ('6', '8'), ('7', '8'), ('8', '9'), ('9', '11'),
             ('9', 'x'), ('14', 'x'), ('x', '12'), ('x', '13')]
    return edges


@pytest.fixture(scope='module')
def many_parents_edges():
    """Node 62 has 18 parents and no children."""
    edges = [('96', '62'), ('80', '62'), ('98', '62'),
             ('100', '62'), ('86', '62'), ('102', '62'), ('104', '62'),
             ('64', '62'), ('106', '62'), ('108', '62'), ('110', '62'),
             ('112', '62'), ('114', '62'), ('116', '62'), ('118', '62'),
             ('122', '62'), ('70', '62'), ('94', '62')]
    return edges


@pytest.fixture(scope='function')
def four_node_model(edges_four_nodes):
    return BernoulliOrModel(edges_four_nodes)


@pytest.fixture(scope='function')
def simple_model(simple_edges):
    return BernoulliOrModel(simple_edges)


@pytest.fixture(scope='function')
def many_parents_model(many_parents_edges):
    return BernoulliOrModel(many_parents_edges)


@pytest.fixture(scope='function')
def one_node_model():
    a_node = BernoulliOrNode(label_id='x', children=set(), parents=set())
    return BernoulliOrModel(edges=None, nodes={'x': a_node})


def get_label_mapped_to_positive_belief(query_result):
    """Return a dictionary mapping each label_id to the probability of
    the label being True."""
    return {label_id: belief[1] for label_id, belief in query_result.items()}


def compare_dictionaries(expected, observed):
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        if observed_value is None:
            raise KeyError("Expected key {} not in observed.")
        assert observed_value == approx(expected_value), \
            "Expected {} but got {}".format(expected_value, observed_value)


#==============================================================================================
# Tests of single Bernoulli node model

def test_no_evidence_one_node_model(one_node_model):
    expected = {'x': 0.5}
    infer = BeliefPropagation(one_node_model)
    query_result = infer.query(evidence={})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


def test_virtual_evidence_one_node_model(one_node_model):
    """Curator thinks YES is 10x more likely than NO based on virtual evidence."""
    expected = {'x': 5/(0.5+5)}
    infer = BeliefPropagation(one_node_model)
    query_result = infer.query(evidence={'x': np.array([1, 10])})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


def test_MAYBE_default_evidence_one_node_model(one_node_model):
    expected = {'x': 0.5}
    infer = BeliefPropagation(one_node_model)
    query_result = infer.query(evidence={'x': np.array([0.5, 0.5])})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


def test_YES_evidence_one_node_model(one_node_model):
    expected = {'x': 1}
    infer = BeliefPropagation(one_node_model)
    query_result = infer.query(evidence={'x': np.array([0, 1])})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


def test_NO_evidence_one_node_model(one_node_model):
    expected = {'x': 0}
    infer = BeliefPropagation(one_node_model)
    query_result = infer.query(evidence={'x': np.array([1, 0])})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


#==============================================================================================
# Tests of 4-node, 4-edge model

def test_no_evidence_four_node_model(four_node_model):
    expected = {'x': 1-0.5**2}
    infer = BeliefPropagation(four_node_model)
    query_result = infer.query(evidence={})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


def test_virtual_evidence_for_node_x_four_node_model(four_node_model):
    """Virtual evidence for node x."""
    expected = {'x': 0.967741935483871, 'y': 0.967741935483871, 'z': 0.967741935483871,
                'u': 0.6451612903225806, 'v': 0.6451612903225806}
    infer = BeliefPropagation(four_node_model)
    query_result = infer.query(evidence={'x': np.array([1, 10])})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


#==============================================================================================
# Tests of simple BernoulliOr polytree model

def test_no_evidence_simple_model(simple_model):
    expected = {'x': 0.984375, '14': 0.5, '7': 0.5, '2': 0.5, '3':
                0.75, '13': 0.984375, '6': 0.5, '4': 0.5, '8': 0.75, '10': 0.875,
                '1': 0.5, '9': 0.96875, '12': 0.984375, '5': 0.875, '11': 0.96875}
    infer = BeliefPropagation(simple_model)
    query_result = infer.query(evidence={})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


def test_belief_propagation_no_modify_model_inplace(simple_model):
    assert not simple_model.all_nodes_are_fully_initialized
    infer = BeliefPropagation(simple_model, inplace=False)
    _ = infer.query(evidence={})
    # after belief propagation, model node values should be unchanged
    assert not simple_model.all_nodes_are_fully_initialized


def test_belief_propagation_modify_model_inplace(simple_model):
    assert not simple_model.all_nodes_are_fully_initialized
    expected = {'x': 0.984375, '14': 0.5, '7': 0.5, '2': 0.5, '3':
                0.75, '13': 0.984375, '6': 0.5, '4': 0.5, '8': 0.75, '10': 0.875,
                '1': 0.5, '9': 0.96875, '12': 0.984375, '5': 0.875, '11': 0.96875}
    infer = BeliefPropagation(simple_model, inplace=True)
    _ = infer.query(evidence={})

    assert simple_model.all_nodes_are_fully_initialized
    beliefs_from_model = {node_id: node.belief[1] for node_id, node in simple_model.nodes.items()}
    compare_dictionaries(expected, beliefs_from_model)


def test_positive_evidence_node_13(simple_model):
    expected = {'6': 0.50793650793650791, '3': 0.76190476190476186,
                '9': 0.98412698412698407, '8': 0.76190476190476186,
                'x': 1.0, '4': 0.50793650793650791, '11': 0.98412698412698407,
                '1': 0.50793650793650791, '5': 0.88888888888888884,
                '2': 0.50793650793650791, '12': 1.0,
                '14': 0.50793650793650791, '13': 1,
                '10': 0.88888888888888884, '7': 0.50793650793650791}
    infer = BeliefPropagation(simple_model)
    query_result = infer.query(evidence={'13': np.array([0, 1])})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


def test_positive_evidence_node_5(simple_model):
    expected = {'1': 0.5714285714285714, '5': 1, '3':
                0.8571428571428571, '10': 1.0, '8': 0.75, '2': 0.5714285714285714,
                '4': 0.5714285714285714, '6': 0.5, '7': 0.5, '14': 0.5, '12': 1.0,
                '13': 1.0, '11': 1.0, '9': 1.0, 'x': 1.0}
    infer = BeliefPropagation(simple_model)
    query_result = infer.query(evidence={'5': np.array([0, 1])})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


def test_positive_evidence_node_5_negative_evidence_node_14(simple_model):
    expected = {'6': 0.5, '7': 0.5, '9': 1.0, '3': 0.8571428571428571,
                '1': 0.57142857142857151, '12': 1.0, 'x': 1.0, '11': 1.0, '14':
                0.0, '2': 0.57142857142857151, '4': 0.5714285714285714, '5': 1.0,
                '10': 1.0, '13': 1.0, '8': 0.75}
    infer = BeliefPropagation(simple_model)
    query_result = infer.query(evidence={'5': np.array([0, 1]), '14': np.array([1, 0])})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


def test_conflicting_evidence(simple_model):
    infer = BeliefPropagation(simple_model)
    with pytest.raises(ConflictingEvidenceError) as err:
        query_result = infer.query(evidence={'x': np.array([1, 0]), '5': np.array([0, 1])})
    assert "Can't run belief propagation with conflicting evidence" in str(err)


#==============================================================================================
# Tests of model with 18 parents sharing a single child

def test_no_evidence_many_parents_model(many_parents_model):
    expected = {'64': 0.5, '86': 0.5, '62': 0.99999618530273438,
                '116': 0.5, '100': 0.5, '108': 0.5, '122': 0.5, '114': 0.5, '98':
                0.5, '106': 0.5, '94': 0.5, '80': 0.5, '102': 0.5, '70': 0.5,
                '118': 0.5, '96': 0.5, '104': 0.5, '110': 0.5, '112': 0.5}
    infer = BeliefPropagation(many_parents_model)
    query_result = infer.query(evidence={})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


def test_positive_evidence_node_112(many_parents_model):
    """If a single parent (112) is True, then (62) has to be True."""
    expected = {'64': 0.5, '86': 0.5, '62': 1.0, '116': 0.5, '100':
                0.5, '108': 0.5, '122': 0.5, '114': 0.5, '98': 0.5,
                '106': 0.5, '94': 0.5, '80': 0.5, '102': 0.5, '70':
                0.5, '118': 0.5, '96': 0.5, '104': 0.5, '110': 0.5,
                '112': 1.0}
    infer = BeliefPropagation(many_parents_model)
    query_result = infer.query(evidence={'112': np.array([0, 1])})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)


def test_negative_evidence_node_62(many_parents_model):
    """If node 62 is False, then all of its parents must be False."""
    expected = {'64': 0, '86': 0, '62': 0, '116': 0, '100': 0, '108':
                0, '122': 0, '114': 0, '98': 0, '106': 0, '94': 0,
                '80': 0, '102': 0, '70': 0, '118': 0, '96': 0, '104':
                0, '110': 0, '112': 0}
    infer = BeliefPropagation(many_parents_model)
    query_result = infer.query(evidence={'62': np.array([1, 0])})
    result = get_label_mapped_to_positive_belief(query_result)
    compare_dictionaries(expected, result)
