#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import numpy as np
import pandas as pd

from . import computation

from .preprocessing import create_user_item_matrix


def mean_rank(
    interactions: pd.DataFrame,
    user_column: str,
    item_column: str,
    feedback_column: str,
    engine: computation.Engine,
) -> np.float:
    """
    Computes mean rank metric using real `interactions`
    as ground truth and predictions produced by `engine`.

    The metric is defined as

    ```
    mean_rank := Σ_ui r_ui * rank_ui / Σ_ui r_ui
    ```

    where `r_ui` is the feedback value between user `u` and item `i`
    and `rank_ui` is the recommendation rank of item `i` for user `u`.

    `rank_ui` is computed by inverse row-wise percentile ranking of
    values in `X * Y^T` prediction matrix. Value `r_ui = 0` means
    that item `i` is the first to be recommended for user `u`,
    `r_uj = 1` is the last to be recommended.

    The metric is a mean rank value weighted by `R` feedbacks.

    Args:
        interactions: user-item feedback pairs
        user_column: name of the column containing user ids
        item_column: name of the column containing item ids
        feedback_column: name of the column containing feedbacks values
        engine: trained model

    Returns:
        computed metric
    """

    # TODO: column/NA checks

    R = create_user_item_matrix(
        interactions=interactions,
        user_column=user_column,
        item_column=item_column,
        feedback_column=feedback_column,
    )

    user_ids, item_ids = R.index.tolist(), R.columns.tolist()

    # prediction = X * Y ^ T
    # ranks = apply rank row wise on prediction
    ranks = (
        engine.user_factors.loc[user_ids, :]
        .dot(engine.item_factors.loc[item_ids, :].T)
        .rank(pct=True, ascending=False, axis=1)
    )

    # multiply ranks with R element-wise, make a sum and divide by R sum
    return R.mul(ranks).sum().sum() / R.sum().sum()
