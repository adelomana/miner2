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


    def test_correct_batch_effects_tpm(self):
        # large means to trigger the TPM function
        df = pd.DataFrame([[4, 1, 2], [1, 2, 3], [4, 5, 6]])
        df2 = preprocess.correct_batch_effects(df)
        self.assertEquals((3, 3), df2.shape)
        self.assertAlmostEquals(df2.values[0, 0], 1.4142135623730949)
        self.assertAlmostEquals(df2.values[1, 0], -1.4142135623730949)
        self.assertAlmostEquals(df2.values[2, 0], -1.4142135623730949)

        self.assertAlmostEquals(df2.values[0, 1], -0.70710678118654746)
        self.assertAlmostEquals(df2.values[1, 1], 0.70710678118654746)
        self.assertAlmostEquals(df2.values[2, 1], 0.70710678118654746)

        self.assertAlmostEquals(df2.values[0, 2], -0.70710678118654746)
        self.assertAlmostEquals(df2.values[1, 2], 0.70710678118654746)
        self.assertAlmostEquals(df2.values[2, 2], 0.70710678118654746)

    def test_correct_batch_effects_no_tpm(self):
        # small means standard deviation
        df = pd.DataFrame([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
        df2 = preprocess.correct_batch_effects(df)
        self.assertEquals((3, 3), df2.shape)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEquals(df2.values[i, j], -0.8164965809277261)

    def test_preprocess_main_simple(self):
        exp, conv_table = preprocess.main('testdata/exp_data-001.csv', 'testdata/conv_table-001.tsv')
        self.assertEquals((10, 3), exp.shape)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEquals(exp.values[i, j], -0.8164965809277261)

if __name__ == '__main__':
    SUITE = []
    SUITE.append(unittest.TestLoader().loadTestsFromTestCase(PreprocessTest))
    if len(sys.argv) > 1 and sys.argv[1] == 'xml':
      xmlrunner.XMLTestRunner(output='test-reports').run(unittest.TestSuite(SUITE))
    else:
      unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(SUITE))
