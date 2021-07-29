#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2020, Jan Cervenka

import multiprocessing as mp
import unittest

from typing import Iterable

import numpy as np
import pandas as pd

from ..core.computation import Engine


class EngineTest(unittest.TestCase):
    """
    Tests `computation.Engine` class.
    """

    def setUp(self) -> None:
        """
        Sets up the tests.
        """

        self._engine = Engine(random_state=0)

    def test_initialize_factors(self) -> None:
        """
        Tests that `Engine._initialize_factors` produce factor
        matrices `X` and `Y` with random values in correct shape.
        """

        test_m, test_n = (15, 25)

        X_expected, Y_expected = self._engine._initialize_factors(test_m, test_n)

        for expected, size in ((X_expected, test_m), (Y_expected, test_n)):
            self.assertTupleEqual((size, self._engine._n_factors), expected.shape)

    def test_feedback_1d_generator(self) -> None:
        """
        Tests that `Engine._feedback_1d_generator` produces correct
        row/column arrays from `R` based on the `axis` argument.
        """

        def assert_collection_of_numpy_equal(
            x: Iterable[np.array], y: Iterable[np.array]
        ) -> None:
            """
            Asserts that collections `x` and `y` containing
            numpy arrays are identical.
            """

            x, y = list(x), list(y)
            assert len(x) == len(x)

            for array_x, array_y in zip(x, y):
                np.testing.assert_array_equal(array_x, array_y)

        test_R = pd.DataFrame({"A": [1, 3], "B": [6, 7]})

        tests = (
            ([np.array([1, 3]), np.array([6, 7])], 0),
            ([np.array([1, 6]), np.array([3, 7])], 1),
        )

        for expected, axis in tests:
            result = list(self._engine._feedback_1d_generator(test_R, axis))
            assert_collection_of_numpy_equal(result, expected)

    def test_compute_factors_1d(self) -> None:
        """
        Tests that `Engine._compute_factors` computes correct
        factor row for one user/item.
        """
        np.random.seed(0)

        test_feedback_1d = np.array([5, 4, 0, 0, 0, 4, 0, 0, 0, 0])
        test_other_factors = np.random.rand(10, 5)
        test_other_factors_small = np.dot(test_other_factors.T, test_other_factors)

        result = self._engine._compute_factors_1d(
            feedback_1d=test_feedback_1d,
            other_factors=test_other_factors,
            other_factors_small=test_other_factors_small,
            reg_lambda=1,
            alpha=40,
        )

        expected = np.array([0.2940538, 0.52162197, 0.82403064, -0.2290155, 0.17128187])

        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    def test_compute_factors(self) -> None:
        """
        Tests that `Engine._compute_factors` computes
        correct factor rows for every user/item.
        """
        np.random.seed(0)

        R_test = pd.DataFrame(
            [
                [3, 2, 0, 0, 0, 0, 0, 0, 1, 2],
                [0, 2, 3, 0, 0, 0, 0, 2, 3, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
                [0, 1, 0, 2, 2, 0, 0, 0, 2, 2],
                [1, 0, 0, 0, 2, 0, 3, 0, 0, 0],
                [1, 0, 0, 0, 2, 3, 0, 3, 0, 0],
                [0, 0, 3, 0, 2, 0, 0, 0, 0, 0],
                [0, 0, 1, 3, 3, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]
        )

        test_n = R_test.shape[-1]
        # axis=1 test for user factors, other_factors matrix is for items
        test_axis = 1
        test_other_factors = np.random.rand(test_n, 3)

        pool = mp.Pool(1)
        try:
            result = self._engine._compute_factors(
                pool=pool, R=R_test, other_factors=test_other_factors, axis=test_axis
            )
        finally:
            pool.close()

        # shape (9, 3): 3 factors for 9 users
        expected = np.array(
            [
                [0.35764017, 1.03013196, 0.27789359],
                [1.21552231, 1.36885415, -0.71711505],
                [1.5414733, -0.52604296, -0.47427519],
                [0.24321584, 1.0214088, 0.40020838],
                [0.04003445, 1.06238047, 0.06082416],
                [-0.51951887, 1.15240956, 1.12119426],
                [-0.79891536, 1.55973535, -0.05766712],
                [-1.09061448, 1.76283214, -0.04719709],
                [0.27658886, 0.94346013, -0.90276183],
            ]
        )

        np.testing.assert_array_almost_equal(result, expected, decimal=7)

    @staticmethod
    def test_predict() -> None:
        """
        Tests that `Engine.predict` computes correct
        recommendation for given `user`.
        """

        engine = Engine()
        engine._X = np.array([[0.50, 0.90, 0.30], [0.05, 0.94, 0.81]])

        engine._Y = np.array(
            [[0.20, 0.12, 0.80], [0.50, 0.97, 0.03], [0.75, 0.02, 0.15]]
        )

        # user 20 likes factors 1 and 2
        # item 4 really belongs to genre 1 and item 3 belongs to genre 2
        # these two items will be recommended
        #
        # user 20 doesnt care about factor 0 where item 5 belongs
        # small recommendation for item 5

        engine._user_index = pd.Index([10, 20], name="user_id")
        engine._item_index = pd.Index([3, 4, 5], name="item_id")

        result = engine.predict(user=20)
        result_top_2 = engine.predict(user=20, top_n=2)

        expected = pd.Series(
            [0.7708, 0.9611, 0.1778], index=engine._item_index, name=20
        )
        # calling .nlargest on series will sort the values (descending)
        expected_top_2 = expected[[True, True, False]].sort_values(ascending=False)

        pd.testing.assert_series_equal(result, expected)
        pd.testing.assert_series_equal(result_top_2, expected_top_2)

    def test_get_loss(self) -> None:
        """
        Tests that `Engine._get_loss` computes correct least squares loss.
        """

        R_test = np.array([[1, 0, 0, 4], [2, 1, 0, 0], [0, 0, 0, 1]])

        X_test = np.array([[0.3, 0.1], [0.9, 0.4], [0.1, 0.5]])

        Y_test = np.array([[0.4, 0.1], [0.9, 0.5], [0.9, 0.9], [0.6, 0.5]])

        result = self._engine._get_loss(X_test, Y_test, R_test)
        expected = 177.73779
        self.assertAlmostEqual(result, expected, places=4)


if __name__ == "__main__":
    unittest.main()
