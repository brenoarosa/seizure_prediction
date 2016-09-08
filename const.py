"""
Defines constant and configurations
"""

import os

CURRENT_DATASET = 1

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

BASEDATA_PATH = CURRENT_PATH + '/dataset/'
TRAIN_DIR = 'train_' + str(CURRENT_DATASET) + '/'
TRAIN_PATH = BASEDATA_PATH + TRAIN_DIR
TEST_DIR = 'test_' + str(CURRENT_DATASET) + '/'
TEST_PATH = BASEDATA_PATH + TEST_DIR

CACHE_PATH = CURRENT_PATH + '/cache/'

FFT_BW = 40
