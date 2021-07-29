#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import warnings

from typing import Tuple, Any

import numpy as np
import pandas as pd


def check_columns_in_dataframe(df: pd.DataFrame, columns: Tuple[str]) -> None:
    """
    Raises an exception if any column name in `columns` is not
    present in `df` dataframe.

    Args:
        df: dataframe to check
        columns: tuple of columns that required to be present
    """

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} is not in the dataframe.")


def check_feedback_column_numeric(
    interactions: pd.DataFrame, feedback_column: str
) -> None:
    """
    Raises an exception if `feedback_column` does not contain numeric values.

    Args:
        interactions: user-item interaction dataframe
        feedback_column: name of the column containing feedback values
    """

    if not pd.api.types.is_numeric_dtype(interactions[feedback_column]):
        raise ValueError(f'Column "{feedback_column}" must be numeric.')


def drop_warn_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops any rows with NA values and raises a warning.

    Args:
        df: dataframe to check

    Returns:
        df without the NA values
    """

    n_0 = len(df)
    df = df.dropna(how="any")

    if len(df) < n_0:
        warnings.warn(
            f"NA values found in the dataframe," f" {n_0 - len(df)} rows removed."
        )

    return df


def cast_numeric_greater_than_zero(
    value: Any, value_name: str, required_type: type
) -> None:
    """
    Checks that `value` is greater than zero and casts
    it to `required_type`.

    Raises an exception `value` not greater than zero.

    Args:
        value: numeric value to check
        value_name: name to be included in the error message
        required_type: target type of the value

    Returns:
        value as required type
    """

    if not isinstance(value, required_type):
        value = required_type(value)

    if value <= 0:
        raise ValueError(f"Value {value_name} must be greater than zero.")

    return value


def get_index_position(index: pd.Index, index_value: Any) -> np.int64:
    """
    Finds position of `index_value` in `index` array.
    This functions is used to find row number of given
    `user_id` or `item_id`.

    Raises an exception if `index` is not unique or the
    value is not found.

    Args:
        index: array of indeces
        index_value: value to find

    Returns:
        position of index_value as an integer
    """

    pos = np.where(index == index_value)[0]
    if len(pos) > 1:
        raise ValueError("Index is not unique.")

    if len(pos) == 0:
        raise ValueError(f"index_value = {index_value} not found in the index.")

    return pos[0]
