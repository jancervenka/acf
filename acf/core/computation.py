#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import logging
import functools
import multiprocessing as mp

from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .preprocessing import create_user_item_matrix
from .utils import (
    cast_numeric_greater_than_zero,
    check_columns_in_dataframe,
    check_feedback_column_numeric,
    get_index_position,
    drop_warn_na,
)

DEFAULT_REG_LAMBDA = 0.1
DEFAULT_ALPHA = 40
DEFAULT_N_FACTORS = 10
DEFAULT_N_ITER = 20
DEFAULT_N_JOBS = 1

# TODO: sparse R dataframe/matrix?
# TODO: X/Y init - which distributions?
# TODO: _compute_factors_1d, _get_least_square_sum to algebra.py module?


class Engine:
    """
    Collaborative filtering algorithm based on matrix factorization
    by alternating least squares. Designed for implicit feedback datasets.

    The class exposes two public methods, `fit` to train the model and
    `predict` to produce the recommendations.

    Example:
    ```
    import acf
    import pandas as pd

    # assuming the data are in the following format:
    # | user_id | item_id | feedback |
    # |---------|---------|----------|
    # | 2491    | 129     | 2        |

    interactions = pd.read_csv('interactions.csv')

    engine = acf.Engine(reg_lambda=1, alpha=35, n_factors=2)
    engine.fit(interactions,
               user_column='user_id',
               item_column='item_id',
               feedback_column='feedback',
               n_iter=20,
               n_jobs=4)

    # get the best 20 recommendation for given user
    prediction = engine.predict(user=2491, top_n=20)

    # to print training loss value at every iteration
    print(engine.loss)
    ```
    """

    def __init__(
        self,
        reg_lambda: float = DEFAULT_REG_LAMBDA,
        alpha: float = DEFAULT_ALPHA,
        n_factors: int = DEFAULT_N_FACTORS,
        random_state: Optional[int] = None,
    ):
        """
        Initializes the class with model hyperparameters
        and `random_state`.

        Args:
            reg_lambda: regularization parameter
            alpha: confidence control parameter
            n_factors: number of latent factors
            random_state: RNG seed
        """

        self._reg_lambda = cast_numeric_greater_than_zero(
            reg_lambda, "reg_lambda", float
        )
        self._alpha = cast_numeric_greater_than_zero(alpha, "alpha", float)
        self._n_factors = cast_numeric_greater_than_zero(n_factors, "n_factors", int)

        if random_state:
            self._rng = np.random.RandomState(random_state)
        else:
            self._rng = np.random.RandomState()

        self._X = self._Y = None
        self._user_index = self._item_index = None
        self._loss = []

    @staticmethod
    def _compute_factors_1d(
        feedback_1d: np.array,
        other_factors: np.array,
        other_factors_small: np.array,
        reg_lambda: float,
        alpha: float,
    ) -> np.array:
        """
        Computes a 1-dimensional factor array for either one user
        or one item.

        When computing user factors, `feedback_1` is an array of interactions
        `r_u` between user `u` and every item, `other_factors` is an item
        factor matrix `Y` and `other_factors_small` is `Y ^ T * Y`.

        When computing the user factors `x_u` for user `u`, the function
        solves a linear system

        ```
        (Y^T * Y + Y^T * (C^u - I) * Y + λ * I) * x_u = Y^T * C^U * p_u
        ```

        where `Y` is an item factor matrix, `C^u` is a diagonal matrix
        containing feedback `r_u` of user `u` for each item `i` (the
        feedback is transformed to `c_ii = 1 + alpha * r_ui`), `p_u`
        is a binary vector of preferences of user `u` for each item,
        and `λ` is regularization lambda.

        The computation is symmetric for item factors.

        Args:
            feedback_1d: one R user-item matrix row or column
            other_factors: either X or Y factor matrix
            other_factors_small: either X^T * X or Y^T * Y matrix
            :reg_lambda: regularization parameter
            alpha: confidence control parameter

        Returns:
            1-dimensional factor array for one user/item
        """
        p_ = (feedback_1d > 0).astype("uint8")
        C_ = np.diag(1 + alpha * feedback_1d)

        size, f = other_factors.shape

        M = other_factors_small + np.linalg.multi_dot(
            [other_factors.T, C_ - np.eye(size), other_factors]
        )

        return np.linalg.solve(
            M + reg_lambda * np.eye(f), np.linalg.multi_dot([other_factors.T, C_, p_])
        )

    def _create_worker(
        self, other_factors: np.array, other_factors_small: np.array
    ) -> Callable:
        """
        Creates a callable worker from `Engine._compute_factors_1d` for
        application over `R` rows/columns in multiprocessing map.

        The worker takes only one argument (`R` column/row), other
        parameters are set in this function using `functools.partial`
        and are identical for all given rows/columns.

        Args:
            other_factors: either X or Y factor matrix
            other_factors_small: either X^T * X or Y^T * Y matrix

        Returns:
            callable worker
        """

        return functools.partial(
            self._compute_factors_1d,
            other_factors=other_factors,
            other_factors_small=other_factors_small,
            reg_lambda=self._reg_lambda,
            alpha=self._alpha,
        )

    @staticmethod
    def _feedback_1d_generator(R: pd.DataFrame, axis: int) -> Iterable[np.array]:
        """
        Creates a generator from `R` rows or columns.
        Each item in the generator is a numpy array.

        Args:
            R: user-item feedback matrix
            axis: axis=1 for rows, axis=0 for columns

        Returns:
            R row/column generator
        """

        for _, r in R.iterrows() if axis else R.iteritems():
            yield r.values

    def _compute_factors(
        self, pool, R: pd.DataFrame, other_factors: np.array, axis: int
    ) -> np.array:
        """
        Applies `Engine._compute_factors_1d` over `R` rows/columns
        to compute factors for all user/items.

        Args:
            pool: multiprocessing pool
            R: user-item feedback matrix
            other_factors: either X or Y factor matrix
            axis: axis=1 for rows, axis=0 for columns

        Returns:
            either X or Y factor matrix
        """
        # TODO: axis can be infered from other_factors shape

        other_factors_small = np.dot(other_factors.T, other_factors)
        worker = self._create_worker(other_factors, other_factors_small)

        all_feedback_1d = self._feedback_1d_generator(R, axis)

        return np.vstack(pool.map(worker, all_feedback_1d))

    def _initialize_factors(self, m: int, n: int) -> Tuple[np.array]:
        """
        Initialzes `X`, `Y` matrices with random values.

        Args:
            m: number of users
            n: number of items

        Returns:
            tuple of X, Y factor matrices
        """

        return (self._rng.rand(m, self._n_factors), self._rng.rand(n, self._n_factors))

    def _get_loss(self, X: np.array, Y: np.array, R: np.array) -> np.float:
        """
        Computes least squares sum for given `X`, `Y`, `R`.

        The loss is defined as `loss := Σ_ui (C * (P - X * Y^T)^2)`
        where `C` is confidence matrix, `P` is preference matrix,
        and `X`, `Y` are factor matrices.

        Args:
            X: user factors matrix
            Y: item factors matrix
            R: user-item feedback matrix

        Returns:
            loss value
        """

        P = (R > 0).astype("uint8")
        C = 1 + self._alpha * R

        return np.sum(C * (P - np.dot(X, Y.T)) ** 2)

    @staticmethod
    def _log_iteration(iteration: int, loss: np.float) -> None:
        """
        Logs least squares `loss` for given `iteration`.

        Args:
            iteration: training iterataion
            loss: value to log
        """

        logging.info(f"Iteration {iteration:04d} Loss Σ = {loss:.5f}")

    def _run_iterations(self, pool, R: pd.DataFrame, n_iter: int) -> Tuple[np.array]:
        """
        Initializes `X`, `Y` matrices and runs
        alternating least squares iterations.

        Args:
            pool: multiprocessing pool
            R: user-item feedback matrix
            n_iter: number of iterations

        Returns:
            tuple of X, Y factor matrices
        """

        X, Y = self._initialize_factors(*R.shape)

        for i in range(n_iter):
            X = self._compute_factors(pool, R, Y, axis=1)  # rows
            Y = self._compute_factors(pool, R, X, axis=0)  # columns
            self._loss.append(self._get_loss(X, Y, R.values))
            self._log_iteration(i, self._loss[-1])

        return X, Y

    def fit(
        self,
        interactions: pd.DataFrame,
        user_column: str,
        item_column: str,
        feedback_column: str,
        n_iter: int = DEFAULT_N_ITER,
        n_jobs: int = DEFAULT_N_JOBS,
    ) -> None:
        """
        Fits the model by factorizing `interactions` into latent factors.

        Dataframe `interactions` contains implicit feedbacks values for
        user-item pairs in three columns `user_column`, `item_column`,
        `feedback_column`.

        ```
        | user_column | item_column | feedback_column    |
        |-------------|-------------|--------------------|
        | user_id_0   | item_id_0   | 0_1_feedback_value |
        ```

        Args:
            interactions: user-item feedback pairs
            user_column: name of the column containing user ids
            item_column: name of the column containing item ids
            feedback_column: name of the column containing feedbacks values
            n_iter: number of alternating least squares iterations
            n_jobs: number of multiprocessing jobs
        """

        check_columns_in_dataframe(
            interactions, (user_column, item_column, feedback_column)
        )
        check_feedback_column_numeric(interactions, feedback_column)

        interactions = drop_warn_na(
            interactions[[user_column, item_column, feedback_column]]
        )

        n_iter = cast_numeric_greater_than_zero(n_iter, "n_iter", int)

        R = create_user_item_matrix(
            interactions=interactions,
            user_column=user_column,
            item_column=item_column,
            feedback_column=feedback_column,
        )

        self._user_index, self._item_index = R.index, R.columns

        pool = mp.Pool(n_jobs)

        try:
            self._X, self._Y = self._run_iterations(pool, R, n_iter)

        finally:
            pool.close()

    def predict(self, user: Any, top_n: Optional[int] = None) -> pd.Series:
        """
        Computes recommendations for given `user`.

        Args:
            user: target of the recommendations
            top_n: if not `None`, selects only the best n items

        Returns:
            series with predicted score for each item
        """

        row = get_index_position(self._user_index, user)

        prediction = pd.Series(
            np.dot(self._Y, self._X[row, :]), index=self._item_index, name=user
        )

        if top_n:
            return prediction.nlargest(top_n)
        else:
            return prediction

    @property
    def user_factors(self) -> pd.DataFrame:
        """
        User factors property.

        Returns:
            user factors as a dataframe
        """

        return pd.DataFrame(self._X, index=self._user_index)

    @property
    def item_factors(self) -> pd.DataFrame:
        """
        Item factors property.

        Returns:
            item factors as a dataframe
        """

        return pd.DataFrame(self._Y, index=self._item_index)

    @property
    def loss(self) -> List[np.float]:
        """
        Training loss proprety.

        Returns:
            training loss by iteration
        """

        return self._loss
