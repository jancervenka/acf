#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import unittest

import numpy as np
import pandas as pd

from ..core import utils


class CheckColumnsInDataFrameTest(unittest.TestCase):
    """
    Tests `utils.check_columns_in_data_frame`.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.
        """

        self._test_df = pd.DataFrame({"x": [1], "y": [1], "z": [1]})

    def test_all_columns_in_dataframe(self) -> None:
        """
        Tests that no exception is raised if a column is present.
        """

        tests = (tuple(), ("x", "y"), ("x",), ("x", "y", "z"))
        for columns in tests:
            utils.check_columns_in_dataframe(self._test_df, columns)

    def test_column_not_in_data_frame(self) -> None:
        """
        Tests that an exception is raised if a column is not present.
        """

        for columns in (("a"), ("x", "a")):
            with self.assertRaises(ValueError):
                utils.check_columns_in_dataframe(self._test_df, columns)


class CheckFeedbackColumnNumericTest(unittest.TestCase):
    """
    Tests `utils.check_feedback_column_numeric`.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.
        """

        self._test_df = pd.DataFrame({"x": ["a", "b"], "y": [1, 3]})

    def test_feedback_column_is_numeric(self) -> None:
        """
        Tests that no exception is raised if the feedback column is numeric.
        """

        utils.check_feedback_column_numeric(self._test_df, "y")

    def test_feedback_column_not_numeric(self) -> None:
        """
        Tests that a exception is raised if the feedback column is not
        numeric.
        """

        with self.assertRaises(ValueError):
            utils.check_feedback_column_numeric(self._test_df, "x")


class DropWarnNaTest(unittest.TestCase):
    """
    Tests `utils.test_drop_warn_na`.
    """

    def test_na_present(self) -> None:
        """
        Tests that rows are dropped and warning is raised when NA
        value is present.
        """

        test_df = pd.DataFrame({"x": [1.0, 2.0, np.nan]})
        expected = pd.DataFrame({"x": [1.0, 2.0]})

        with self.assertWarns(UserWarning):
            result = utils.drop_warn_na(test_df)

        pd.testing.assert_frame_equal(result, expected)

    @staticmethod
    def test_na_not_present() -> None:
        """
        Tests that original dataframe is returned when no NA is present.
        """

        test_df = expected = pd.DataFrame({"x": [1, 2]})
        expected = pd.DataFrame({"x": [1, 2]})

        result = utils.drop_warn_na(test_df)
        pd.testing.assert_frame_equal(result, expected)


class CastNumericGreaterThanZeroTest(unittest.TestCase):
    """
    Tests `utils.test_drop_warn_na`.
    """

    def test_value_converted(self) -> None:
        """
        Tests that compatible `value` are correctly cast to `required_type`.
        """

        tests = ((1, "test", int, 1), (2.3, "test", float, 2.3), (1.2, "test", int, 1))

        for value, value_name, required_type, expected in tests:
            result = utils.cast_numeric_greater_than_zero(
                value, value_name, required_type
            )

            self.assertEqual(result, expected)

    def test_value_not_numeric(self) -> None:
        """
        Test that an exception is raised when a non-numeric `value`
        cannot be cast to `required_type`.
        """

        with self.assertRaises(ValueError):
            _ = utils.cast_numeric_greater_than_zero("test", "test", float)

    def test_value_not_greater_than_zero(self) -> None:
        """
        Test that an exception is raised when a numeric `value`
        is not greater than zero.
        """

        with self.assertRaises(ValueError):
            _ = utils.cast_numeric_greater_than_zero(-1, "test", float)


class GetIndexPositionTest(unittest.TestCase):
    """
    Tests `utils.get_index_position`.
    """

    def test_index_value_not_unique(self) -> None:
        """
        Tests that an exception is raised when
        `index_value` is not unique in `index`.
        """

        test_index = pd.Index([1, 3, 2, 2])
        test_index_value = 2

        with self.assertRaises(ValueError):
            _ = utils.get_index_position(test_index, test_index_value)

    def test_index_value_not_found(self) -> None:
        """
        Tests that an exception is raised when
        `index_value` is not present in `index`.
        """

        test_index = pd.Index([1, 3, 2])
        test_index_value = 4

        with self.assertRaises(ValueError):
            _ = utils.get_index_position(test_index, test_index_value)

    def test_index_value_found(self) -> None:
        """
        Tests that correct row positon is returned.
        """

        test_index = pd.Index([1, 3, 2, 4])
        test_index_value = 3

        result = utils.get_index_position(test_index, test_index_value)
        expected = 1

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
