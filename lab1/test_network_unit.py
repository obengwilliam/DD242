"""Test for Network."""
from network import load_batch, normalize, normalize_mean_std, Network
import numpy as np
import unittest


class CustomAssertions:
    def assertAlmostEqualArray(self, arr1, arr2, dec=2):
        np.testing.assert_almost_equal(arr1, arr2, decimal=dec)

    def assertAlmostEqualArrays(self, arr1, arr2, dec=2):
        np.testing.assert_almost_equal(arr1, arr2, decimal=dec)


class TestNetwork(unittest.TestCase, CustomAssertions):
    X_train, Y_train, y_train = load_batch("data_batch_1")

    X_train, X_train_mean, X_train_std = normalize(X_train)

    X_test, Y_test, y_test = load_batch("test_batch")

    X_test = normalize_mean_std(X_test, X_train_mean, X_train_std)

    X_val, Y_val, y_val = load_batch(("data_batch_2"))
    X_val = normalize_mean_std(X_val, X_train_mean, X_train_std)

    data = {
        "X_train": X_train,
        "Y_train": Y_train,
        "y_train": y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "y_test": y_test,
        "X_val": X_val,
        "Y_val": Y_val,
        "y_val": y_val,
    }

    network = Network(data)

    def test_shapes(self):
        """Test shapes all the train test and validation data set."""

        self.assertEqual(np.shape(self.network.Y), (10, 10000))
        self.assertEqual(np.shape(self.network.X), (3072, 10000))
        self.assertEqual(np.shape(self.network.y), (10000,))
        self.assertEqual(
            np.shape(self.network.evaluate_classifier(self.data["X_train"])), (10, 10000)
        )

    def test_array_equality(self):
        """Validate difference between numerical and analytical gradients."""

        grad_W_test_num, grad_b_test_num = self.network.compute_gradients_num(
            self.network.X[:, :1], self.network.Y[:, :1], lamda=0
        )
        grad_W_test_ana, grad_b_test_ana = self.network.compute_gradients_ana(
            self.network.X[:, :1], self.network.Y[:, :1], lamda=0
        )

        errors = np.abs(grad_W_test_num - grad_W_test_ana)
        max_values = np.full(grad_W_test_ana.shape, 1e-6)
        den = np.maximum(max_values, np.abs(grad_W_test_ana) + np.abs(grad_W_test_num))

        rel_errors = errors / den

        self.assertTrue(np.all(rel_errors < 1e-03), "Some are less than 1e-03")
        num_less = np.sum(rel_errors > 1e-04)
        print(num_less)
        # randomly select 10 relative errors and make sure they are not more than
        # print(rel_errors.shape)


#         self.assertAlmostEqualArrays(grad_W_test_ana, grad_W_test_num, 4)
#         self.assertAlmostEqualArrays(grad_b_test_ana, grad_b_test_num, 4)
