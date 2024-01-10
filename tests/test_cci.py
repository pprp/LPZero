import unittest
from unittest import TestCase
from lpzero.metrics.cluster_correlation_index import measure_cluster_corr_index
import numpy as np

class TestCCITestCase(TestCase):
    def test_cci(self):
        var1 = np.random.randn(10)
        var2 = np.random.randn(10)
        n_clusters = 3
        cci = measure_cluster_corr_index(var1, var2, n_clusters)
        print(cci)

if __name__ == '__main__':
    unittest.main()        