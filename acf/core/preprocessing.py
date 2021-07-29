#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import pandas as pd


def create_user_item_matrix(
    interactions: pd.DataFrame, user_column: str, item_column: str, feedback_column: str
) -> pd.DataFrame:
    """
    Creates `R` from `interactions` dataframe by pivoting it
    from long to wide format.

    Args:
        interactions: user-item feedback pairs
        user_column: name of the column containing user ids
        item_column: name of the column containing item ids
        feedback_column: name of the column containing feedbacks values

    Returns:
        R user-item matrix
    """

    # handles user-item duplicates by suming the feedback
    return interactions.pivot_table(
        index=user_column, columns=item_column, aggfunc="sum", values=feedback_column
    ).fillna(0)
