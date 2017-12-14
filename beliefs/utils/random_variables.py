"""Utilities for working with models and random variables."""


def get_reachable_observed_variables_for_inferred_variables(model, observed=set()):
    """
    After performing inference on a BayesianModel, get the labels of observed variables
    ("reachable observed variables") that influenced the beliefs of variables inferred
    to be in a definite state.

    Args
        model: instance of BayesianModel class or subclass
        observed: set,
            set of labels (strings) corresponding to variables pinned to a definite
            state during inference.
    Returns
        dict,
            key, value pairs {source_label_id: reachable_observed_vars}, where
            source_label_id is an int or string, and reachable_observed_vars is a list
            of label_ids
    """
    if not observed:
        return {}

    source_vars = model.get_unobserved_variables_in_definite_state(observed)

    return {var: model.reachable_observed_variables(var, observed) for var in source_vars}
