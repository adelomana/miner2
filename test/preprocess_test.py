#!/usr/bin/env python3
import sys
import unittest

import pandas as pd
from miner2 import preprocess

class PreprocessTest(unittest.TestCase):

    def test_remove_null_rows_min_0_remove_ok(self):
        df = pd.DataFrame([[0, 1, 2], [1, 2, 3], [0, 0, 0], [4, 5, 6]])
        df2 = preprocess.remove_null_rows(df)
        self.assertEqual(3, df2.shape[0], "wrong number of rows")

    def test_remove_null_rows_min_0_unchanged(self):
        df = pd.DataFrame([[0, 1, 2], [1, 2, 3], [1, 0, 1], [4, 5, 6]])
        df2 = preprocess.remove_null_rows(df)
        self.assertEqual(4, df2.shape[0], "wrong number of rows")

    def test_remove_null_rows_min_negative_unchanged(self):
        df = pd.DataFrame([[0, 1, -2], [1, 2, 3], [0, 0, 0], [4, 5, 6]])
        df2 = preprocess.remove_null_rows(df)
        self.assertEqual(4, df2.shape[0], "wrong number of rows")

if __name__ == '__main__':
    SUITE = []
    SUITE.append(unittest.TestLoader().loadTestsFromTestCase(PreprocessTest))
    if len(sys.argv) > 1 and sys.argv[1] == 'xml':
      xmlrunner.XMLTestRunner(output='test-reports').run(unittest.TestSuite(SUITE))
    else:
      unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(SUITE))
