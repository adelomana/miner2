#!/bin/bash

PYTHONPATH=. python3 test/preprocess_test.py
PYTHONPATH=. python3 test/coexpression_test.py
