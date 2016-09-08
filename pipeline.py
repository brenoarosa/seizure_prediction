"""
Controls the dataflow
"""

import os
import logging
import numpy as np
from data_plot import plot_time_series, plot_fft
from handler import load_train, load_test, save
from const import *

# Logging conf
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s\t%(levelname)s: %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

def main():
    logger.info('Starting pipeline!')

    logger.info('Loading X and y.')
    X, y = load_train()

    logger.info('Saving X, y into cache.')
    save(X, filename='X.p')
    save(y, filename='y.p')

    logger.info('Loading X_test.')
    X_test = load_test()

    logger.info('Saving X_test into cache.')
    save(X_test, filename='X_test.p')

if __name__ == '__main__':
    main()
