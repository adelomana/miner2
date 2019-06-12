#!/usr/bin/env python3
import sys
import unittest

import pandas as pd
from miner2 import preprocess
from miner2 import coexpression

class CoexpressionTest(unittest.TestCase):

    def test_cluster(self):
        exp, conv_table = preprocess.main('testdata/exp_data-002.csv', 'testdata/conv_table-002.tsv')
        expected_clusters = []
        with open('testdata/expected_clusters-002.csv', 'r') as infile:
            for line in infile:
                expected_clusters.append(line.strip().split(','))
        clusters = coexpression.cluster(exp)
        self.assertEquals(7, len(clusters))
        self.assertEquals(expected_clusters, clusters)


if __name__ == '__main__':
    SUITE = []
    SUITE.append(unittest.TestLoader().loadTestsFromTestCase(CoexpressionTest))
    if len(sys.argv) > 1 and sys.argv[1] == 'xml':
      xmlrunner.XMLTestRunner(output='test-reports').run(unittest.TestSuite(SUITE))
    else:
      unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(SUITE))
