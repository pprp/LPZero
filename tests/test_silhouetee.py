import unittest
from unittest import TestCase

import numpy as np

from lpzero.metrics.silhouetee import measure_silhouette


class TestSilhouetee(TestCase):
    def test_silhouetee(self):
        var1 = np.random.randn(10)
        var2 = np.random.randn(10)
        n_clusters = 3
        s_avg = measure_silhouette(var1, var2, n_clusters)
        print(s_avg)


if __name__ == '__main__':
    unittest.main()
