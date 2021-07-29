#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import unittest
from unittest import mock

import pandas as pd

from ..core.metrics import mean_rank


class MeanRankTest(unittest.TestCase):
    """
    Tests `metrics.mean_rank`.
    """

    def test_mean_rank_correct_value(self) -> None:
        """
        Tests that correct mean rank value is computed based
        on `interactions` and `X`, `Y` factors.
        """

        engine = mock.Mock()

        engine.user_factors = pd.DataFrame(
            {0: [0.5, 0.1, 0.9], 1: [0.1, 0.2, 0.5]}, index=["u1", "u2", "u3"]
        )

        engine.item_factors = pd.DataFrame(
            {0: [0.1, 0.2, 0.3, 0.3], 1: [0.5, 0.9, 0.9, 0.7]},
            index=["i1", "i2", "i3", "i4"],
        )

        test_interactions = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u3", "u3", "u3"],
                "item_id": ["i1", "i2", "i4", "i2", "i3", "i1", "i2", "i3", "i4"],
                "feedback": [4, 2, 1, 5, 10, 9, 8, 12, 8],
            }
        )

        # R =
        # 4 2  0 1
        # 0 5 10 0
        # 9 8 12 8

        # X * Y^T =
        # 1.00 0.75 0.25 0.50
        # 0.11 0.20 0.21 0.17
        # ą0.34 0.63 0.72 0.62

        # X * Y^T ranks
        # 1.00 0.75 0.25 0.50
        # 1.00 0.50 0.25 0.75
        # 1.00 0.50 0.25 0.75

        result = mean_rank(
            interactions=test_interactions,
            user_column="user_id",
            item_column="item_id",
            feedback_column="feedback",
            engine=engine,
        )

        expected = 33 / 59
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
