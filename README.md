# Acf

[1]: https://dl.acm.org/doi/10.1109/ICDM.2008.22

*A lightweight recommender engine for implicit feedback datasets*

![PyPI](https://badge.fury.io/py/acf.svg)
![Test](https://github.com/jancervenka/acf/actions/workflows/test.yml/badge.svg)
![Publish](https://github.com/jancervenka/acf/actions/workflows/publish.yml/badge.svg)

The package implements an algorithm described in
[Collaborative Filtering for Implicit Feedback Datasets][1] paper. 
The algorithm is based on the following ideas:

* using collaborative filtering with latent factors
* transforming feedback observations into binary preferences
  with associated confidence levels
* using alternating least sqaures to compute the matrix factorization

## Install

The package requires Python `3.7` or newer, the only dependencies are
`numpy` and `pandas`. To install it, run

```bash
pip install acf
```

## Usage

The following example shows how to train a model and compute predictions.

```python
import acf
import pandas as pd

# assuming the data are in the following format:
# | user_id | item_id | feedback |
# |---------|---------|----------|
# | 2491    | 129     | 2        |

interactions = pd.read_csv('interactions.csv')

engine = acf.Engine(reg_lambda=1, alpha=35, n_factors=2, random_state=0)

engine.fit(interactions,
           user_column='user_id',
           item_column='item_id',
           feedback_column='feedback',
           n_iter=20,
           n_jobs=4)

# get the best 20 recommendations
prediction = engine.predict(user=2491, top_n=20)

# to print training loss value at every iteration
print(engine.loss)
```

### Model Evaluation

For performance evaluation, the package offers `metrics.mean_rank`
function that implements "mean rank" metric as defined by equation
8 in the [paper][1].

The metric is a weighted mean of percentile-ranked recommendations
(`rank_ui = 0` says that item `i` is the first to be recommended for
user `u` and item `j` with `rank_uj = 1` is the last to be recommended)
where the weights are the actual feedback values from `R` user-item matrix.

```python
interactions_test = pd.read_csv('intercations_test.csv')

print(acf.metrics.mean_rank(interactions=interactions_test,
                            user_column='user_id',
                            item_column='item_id'
                            feedback_column='feedback',
                            engine=engine))
```

### Model Persistence

Trained model can be serialized and stored using `joblib` or `pickle`.

To store a model:

```python
with open('engine.joblib', 'wb') as f:
    joblib.dump(engine, f)
```

To load a model:

```python
with open('engine.joblib', 'rb') as f:
    engine = joblib.load(f)
```

## Public API

### `acf.Engine`

```python
acf.core.computation.Engine(reg_lambda=0.1, alpha=40,
                            n_factors=10, random_state=None):
```

Class exposing the recommender.

* `reg_lambda`: regularization strength
* `alpha`: gain parameter in feedback-confidence transformation 
           `c_ui = 1 + alpha * r_ui`
* `n_factors`: number of latent factors
* `random_state`: initial RNG state

__Properties:__

* `user_factors`: user factor matrix
* `item_factors`: item factor matrix
* `loss`: training loss history

__Methods:__

```python
Engine.fit(interactions, user_column, item_column,
           feedback_column, n_iter=20, n_jobs=1)
```

Trains the model.

* `interactions`: dataframe containing user-item feedbacks
* `user_column`: name of the column containing user ids
* `item_column`: name of the column containing item ids
* `feedback_column`: name of the column containing feedback values
* `n_iter`: number of alternating least squares iteration
* `n_jobs`: number of parallel jobs

```python
Engine.predict(user, top_n=None)
```

Predicts the recommendation.

* `user`: user identification for whom the prediction is computed
* `top_n`: if not `None`, only the besr n items are included in the result

__Returns:__ predicted recommendation score for each item as `pandas.Series`

### `acf.metrics.mean_rank`

```python
acf.core.metrics.mean_rank(interactions, user_column, item_column,
                           feedback_column, engine)
```

Computes mean rank evaluation.

* `interactions`: dataframe containing user-item feedbacks
* `user_column`: name of the column containing user ids
* `item_column`: name of the column containing item ids
* `feedback_column`: name of the column containing feedback values
* `engine`: trained `acf.Engine` instance

__Returns:__ computed value

## Tests

Tests can be executed by `pytest` as

```bash
python -m pytest acf/tests
```
