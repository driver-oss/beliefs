# beliefs

A library for performing inference with Bayesian Networks for a
special use case, derived
from [pgmpy](https://github.com/pgmpy/pgmpy).


## Motivation

**Exact inference**

This library provides the ability to perform exact inference in a
computationally tractable* way for a specific but useful case: Bayesian
Networks with
* polytree structure
* consisting of Bernoulli random variables whose relationship to their
  parents in the probabilistic graphical model are described by AND or
  OR logic

Non-deterministic conditional probability distributions for
multinomial, discrete random variables are also supported, although
the algorithm is specifically optimized for the case of Bernoulli AND
and Bernoulli OR variables.

*See the "Many parents model" in
    the
    [jupyter notebook](https://render.githubusercontent.com/view/ipynb?commit=73aa4a35d08f1c16569bc78d176710381b9e9605&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f64726976657267726f75702f62656c696566732f373361613461333564303866316331363536396263373864313736373130333831623965393630352f6578616d706c65732f636f6d706172655f70676d70795f62656c6965665f70726f7061676174696f6e2e6970796e623f746f6b656e3d4158386c536f5a35622d2d7848394a736a58727a65345a7846587531333150426b733561646f4f567741253344253344&nwo=drivergroup%2Fbeliefs&path=examples%2Fcompare_pgmpy_belief_propagation.ipynb&repository_id=110306600&repository_type=Repository#IV.-Many-parents-model) under
    the examples/ directory for an example of a case in which
    inference becomes computationally intractable with pgmpy but can
    be handled by beliefs optimized algorithm.

## Additional features

* In addition to being able to perform inference based on direct
  observation of a variable in the PGM, beliefs also provides the
  ability to specify virtual evidence, i.e. evidence that modifies the
  belief, or marginal probability, of a variable by affecting its
  likelihood based on observations of variables not in the PGM, while
  not pinning the variable into a definite (observed) state.
* The ability to catch conflicting evidence errors during inference,
  which manifest as numpy NaNs in pgmpy's inference results.
* Utility for gathering the direct observations that influenced the
  beliefs of variables that were inferred to be in a definite state.


## Getting started

### Installation

Using conda:
```
conda install -c driver beliefs
```

### Example

Perform inference on a Bayesian Network:
```python
from beliefs.inference.belief_propagation import BeliefPropagation
from beliefs.models.belief_update_node_model import (
    BeliefUpdateNodeModel,
    BernoulliOrNode
)

# directed edges for a polytree Bayes Net
edges = [('u', 'x'), ('v', 'x'), ('x', 'y'), ('x', 'z')

# initialize model w/ edges, default to OR CPD for all variables
model = BeliefUpdateNodeModel.init_from_edges(edges, BernoulliOrNode)

# initialize inference
infer = BeliefPropagation(model)

# perform inference, with 'x' is observed to be True.
result = infer.query(evidence={'x': np.array([0, 1])})
```

## Tests

From the project root directory:
```
pytest tests -vv
```

## License
This project is licensed under the terms of the MIT license.
