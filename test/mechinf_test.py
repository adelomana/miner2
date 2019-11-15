#!/usr/bin/env python3
import sys
import unittest

import pandas as pd
import logging
import json

from miner2 import mechanistic_inference as mechinf

MIN_REGULON_GENES = 5

class MechinfTest(unittest.TestCase):

    def compare_dicts(self, d1, d2):
        """compare 1-level deep dictionary"""
        ref_keys = sorted(d1.keys())
        keys = sorted(d2.keys())
        self.assertEquals(ref_keys, keys)

        for key in keys:
            ref_genes = sorted(d1[key])
            genes = sorted(d2[key])
            self.assertEquals(ref_genes, genes)

    def compare_dicts2(self, d1, d2):
        """compare 2-level deep dictionary"""
        ref_keys = sorted(d1.keys())
        keys = sorted(d2.keys())
        self.assertEquals(ref_keys, keys)

        for key1 in keys:
            # note that the keys from the JSON file has string-values integers as keys
            # while the keys2 are integer keys !!! This is regardless whether Python2 or 3
            ref_keys2 = sorted(map(int, d1[key1].keys()))
            keys2 = sorted(d2[key1].keys())
            self.assertEquals(ref_keys2, keys2)

            # compare gene lists on the second level
            for key2 in keys2:
                ref_genes = sorted(d1[key1][str(key2)])  # note that the ref dict contains string keys
                genes = sorted(d2[key1][key2])
                self.assertEquals(ref_genes, genes)

    def test_get_coexpression_modules(self):
        # test data was generated with the Python 2 version
        with open('testdata/mechanisticOutput-001.json') as infile:
            mechout = json.load(infile)
        with open('testdata/coexpressionModules-001.json') as infile:
            ref_coexp_mods = json.load(infile)
        coexp_mods = mechinf.get_coexpression_modules(mechout)
        self.compare_dicts(ref_coexp_mods, coexp_mods)

    def test_get_coregulation_modules(self):
        with open('testdata/mechanisticOutput-001.json') as infile:
            mechout = json.load(infile)
        with open('testdata/coregulationModules-001.json') as infile:
            ref_coreg_mods = json.load(infile)

        coreg_mods = mechinf.get_coregulation_modules(mechout)
        self.compare_dicts(ref_coreg_mods, coreg_mods)

    # TODO: break down the function to see the differences
    """
    def test_get_regulons(self):
        with open('testdata/ref_regulons-001.json') as infile:
            ref_regulons = json.load(infile)
        with open('testdata/coregulationModules-001.json') as infile:
            coreg_mods = json.load(infile)

        regulons = mechinf.get_regulons(coreg_mods,
                                        min_number_genes=MIN_REGULON_GENES,
                                        freq_threshold=0.333)
        self.compare_dicts2(ref_regulons, regulons)"""

    def test_coincidence_matrix(self):
        with open('testdata/sub_regulons-001.json') as infile:
            sub_regulons = json.load(infile)
        ref_norm_df = pd.read_csv('testdata/coincidence_matrix-001.csv', index_col=0, header=0)
        norm_df = mechinf.coincidence_matrix(sub_regulons, 0.333)
        self.assertTrue(ref_norm_df.equals(norm_df))


if __name__ == '__main__':
    SUITE = []
    LOG_FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S \t')
    SUITE.append(unittest.TestLoader().loadTestsFromTestCase(MechinfTest))
    if len(sys.argv) > 1 and sys.argv[1] == 'xml':
      xmlrunner.XMLTestRunner(output='test-reports').run(unittest.TestSuite(SUITE))
    else:
      unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(SUITE))
