

def get_reachable_observed_variables_for_inferred_variables(model, observed=set()):
    """
    After performing inference on a BayesianModel, get the labels of observed variables
    ("reachable observed variables") that influenced the beliefs of variables inferred
    to be in a definite state.

    INPUT
      model: instance of BayesianModel class or subclass
      observed: set of labels (strings) corresponding to vars pinned to definite
        state during inference.
    RETURNS
      dict, of form key - source label (a string), value - a list of strings
    """
    if not observed:
        return {}

    source_vars = model.get_unobserved_variables_in_definite_state(observed)

    return {var: model.reachable_observed_variables(var, observed) for var in source_vars}
