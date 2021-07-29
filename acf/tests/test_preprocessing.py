#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import unittest

import numpy as np
import pandas as pd

from ..core.preprocessing import create_user_item_matrix


class CreateUserItemMatrixTest(unittest.TestCase):
    """
    Tests `preprocessing.create_user_item_matrix`.
    """

    @staticmethod
    def test_create_user_item_matrix() -> None:
        """
        Tests that the created user-item matrix is correct and
        that duplicate user-item pairs are correctly aggregated.
        """

        test_interactions = pd.DataFrame(
            {
                "id_item": ["A", "B", "C", "A", "B", "A", "D", "D"],
                "id_user": [1, 1, 1, 2, 2, 3, 4, 4],
                "feedback": [6, 2, 1, 5, 4, 2, 1, 2],
            }
        )

        result = create_user_item_matrix(
            test_interactions, "id_user", "id_item", "feedback"
        )

        expected_matrix = np.array(
            [[6, 2, 1, 0], [5, 4, 0, 0], [2, 0, 0, 0], [0, 0, 0, 3]], dtype="float64"
        )

        expected = pd.DataFrame(
            expected_matrix,
            index=pd.Index([1, 2, 3, 4], name="id_user"),
            columns=pd.Index(["A", "B", "C", "D"], name="id_item"),
        )

        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
