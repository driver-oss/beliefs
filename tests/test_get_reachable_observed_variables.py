import numpy as np

from test_belief_propagation import simple_model, simple_edges

from beliefs.inference.belief_propagation import BeliefPropagation
from beliefs.utils.random_variables import (
    get_reachable_observed_variables_for_inferred_variables
)


def test_reachable_observed_vars_direct_common_effect(simple_model):
    observed_vars = {'14': np.array([1,0]), 'x': np.array([1,0])}
    infer = BeliefPropagation(simple_model)
    infer.query(evidence=observed_vars)

    expected = {'x', '14'}
    observed = simple_model.reachable_observed_variables(
        source='9',
        observed=set(observed_vars.keys())
    )
    assert expected == observed


def test_reachable_observed_vars_indirect_common_effect(simple_model):
    observed_vars = {'12': np.array([1,0]), '14': np.array([1,0])}
    infer = BeliefPropagation(simple_model)
    infer.query(evidence=observed_vars)

    expected = {'12', '14'}
    observed = simple_model.reachable_observed_variables(
        source='9',
        observed=set(observed_vars.keys())
    )
    assert expected == observed


def test_reachable_observed_vars_common_cause(simple_model):
    observed_vars = {'10': np.array([0,1])}
    infer = BeliefPropagation(simple_model)
    infer.query(evidence=observed_vars)

    expected = {'10'}
    observed = simple_model.reachable_observed_variables(
        source='9',
        observed=set(observed_vars.keys())
    )
    assert expected == observed


def test_reachable_observed_vars_blocked_common_cause(simple_model):
    observed_vars = {'10': np.array([0,1]), '5': np.array([0,1])}
    infer = BeliefPropagation(simple_model)
    infer.query(evidence=observed_vars)

    expected = {'5'}
    observed = simple_model.reachable_observed_variables(
        source='9',
        observed=set(observed_vars.keys())
    )
    assert expected == observed


def test_reachable_observed_vars_indirect_causal(simple_model):
    observed_vars = {'1': np.array([0,1]), '2': np.array([1,0])}
    infer = BeliefPropagation(simple_model)
    infer.query(evidence=observed_vars)

    expected = {'1', '2'}
    observed = simple_model.reachable_observed_variables(
        source='9',
        observed=set(observed_vars.keys())
    )
    assert expected == observed


def test_reachable_observed_vars_blocked_causal(simple_model):
    observed_vars = {'1': np.array([0,1]), '2': np.array([1,0]), '3': np.array([0,1])}
    infer = BeliefPropagation(simple_model)
    infer.query(evidence=observed_vars)

    expected = {'3'}
    observed = simple_model.reachable_observed_variables(
        source='9',
        observed=set(observed_vars.keys())
    )
    assert expected == observed


def test_reachable_observed_vars_indirect_evidential(simple_model):
    observed_vars = {'13': np.array([1,0])}
    infer = BeliefPropagation(simple_model)
    infer.query(evidence=observed_vars)

    expected = {'13'}
    observed = simple_model.reachable_observed_variables(
        source='9',
        observed=set(observed_vars.keys())
    )
    assert expected == observed


def test_reachable_observed_vars_blocked_evidential(simple_model):
    observed_vars = {'x': np.array([1,0]), '13': np.array([1,0])}
    infer = BeliefPropagation(simple_model)
    infer.query(evidence=observed_vars)

    expected = {'x'}
    observed = simple_model.reachable_observed_variables(
        source='9',
        observed=set(observed_vars.keys())
    )
    assert expected == observed


def test_get_reachable_obs_vars_for_inferred(simple_model):
    observed_vars = {'6': np.array([1,0]), '7': np.array([1,0]), '10': np.array([1,0])}
    infer = BeliefPropagation(simple_model)
    infer.query(evidence=observed_vars)

    print(set(simple_model.get_unobserved_variables_in_definite_state(observed_vars.keys())))
    print(simple_model._get_ancestors_of(set(observed_vars.keys())))
    expected = {'4': {'10'}, '1': {'10'}, '11': {'7', '6', '10'}, '2': {'10'},
                '8': {'7', '6'}, '5': {'10'}, '3': {'10'}, '9': {'7', '6', '10'}}

    observed = get_reachable_observed_variables_for_inferred_variables(
        model=simple_model,
        observed=set(observed_vars.keys())
    )
    assert expected == observed
