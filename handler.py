import os
import pickle
import logging
import numpy as np
from scipy.io import loadmat
from feature_extraction import feature_extraction
from const import *


logger = logging.getLogger('kaggle_seizure')

def load_matdata(filepath):
    return loadmat(filepath)


def load_train(path=TRAIN_PATH):

    files = []

    for f in os.listdir(path):
        if not os.path.isfile(os.path.join(path, f)):
            continue

        f_desc = f.split('_')

        if f_desc[2] == '0.mat':
            files.append(((path + f), 0))
        elif f_desc[2] == '1.mat':
            files.append(((path + f), 1))
        else:
            raise Exception('Invalid filename')

    n_examples = 0
    X = np.array([])
    y = []
    for f in files:
        filepath = f[0]
        cls = int(f[1])

        try:
            matdata = load_matdata(filepath)
        except Exception as e:
            logger.error("Ignoring corruped file: {}".format(filepath))
            continue

        logger.debug("Extracting features from {}: {}".format(n_examples, filepath))

        x = feature_extraction(matdata)

        flat = x.flatten()
        X = np.hstack((X, flat))
        y.append(cls)

        n_feat = len(flat)
        n_examples = n_examples + 1

    logger.debug('{} examples loaded.'.format(n_examples))
    X = X.reshape(n_examples, n_feat)
    y = np.array(y)
    return X, y


def load_test(path=TEST_PATH):

    files = []

    for f in os.listdir(path):
        if not os.path.isfile(os.path.join(path, f)):
            continue

        files.append((path + f))


    n_examples = 0
    X_test = np.array([])
    for f in files:
        filepath = f

        try:
            matdata = load_matdata(filepath)
        except Exception as e:
            logger.error("Ignoring corruped file: {}".format(filepath))
            continue

        logger.debug("Extracting features from {}: {}".format(n_examples, filepath))

        x = feature_extraction(matdata)

        flat = x.flatten()
        X_test = np.hstack((X_test, flat))

        n_feat = len(flat)
        n_examples = n_examples + 1

    logger.debug('{} examples loaded.'.format(n_examples))
    X_test = X_test.reshape(n_examples, n_feat)
    return X_test


def save(data, filename='unnamed.p'):
    filepath = os.path.join(CACHE_PATH, filename)
    f = open(filepath, 'wb')
    pickle.dump(data, f)
    f.close()

def load(filename='unnamed.p'):
    filepath = os.path.join(CACHE_PATH, filename)
    f = open(filepath, 'rb')
    data = pickle.load(f)
    f.close()
    return data
