import unittest
import numpy as np
from lpzero.metrics.mutual_info import measure_mutual_information

class TestMeasureMutualInformation(unittest.TestCase):
    def test_mutual_information_independent_variables(self):
        # Independent variables should have a mutual information close to 0 but not exactly 0 due to noise
        # Increasing sample size for a better estimate
        np.random.seed(42)
        var1 = np.random.rand(1000)
        var2 = np.random.rand(1000)
        mi_score = measure_mutual_information(var1, var2)
        print(mi_score)

    def test_mutual_information_identical_variables(self):
        # Identical variables should have high mutual information
        np.random.seed(42)
        var1 = np.random.rand(1000)
        mi_score = measure_mutual_information(var1, var1)
        self.assertGreater(mi_score, 0.9)  # Expecting a high value, close to 1

    def test_mutual_information_linear_relation(self):
        # Linearly related variables should have a mutual information greater than independent
        np.random.seed(42)
        var1 = np.random.rand(1000)
        var2 = var1 * 2 + 1  # Perfect linear relationship
        mi_score = measure_mutual_information(var1, var2)
        self.assertGreater(mi_score, 0.1)  # Expecting a value greater than for independent variables

if __name__ == '__main__':
    unittest.main()