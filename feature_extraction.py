"""
Extracts features from data
"""
import logging
import numpy as np
from const import *

logger = logging.getLogger(__name__)

def fft(matdata):
    """
    Return fft from 0 to FFT_BW hz
    """

    # Total Samples
    n_samples = matdata["dataStruct"]["nSamplesSegment"][0][0][0][0]
    # Sampling Rate
    fs = matdata["dataStruct"]['iEEGsamplingRate'][0][0][0][0]
    # [n_examples, n_electrodes]
    time_data = matdata['dataStruct']['data'][0][0]

    # Cutting bandwidth
    bandwidth = FFT_BW

    # Runs FFT for real value data
    freq_data = np.fft.rfft(time_data, axis=0)

    # Low-pass cut to dimension reduction
    freq_data = freq_data[0:bandwidth, :]

    # Ignores phase
    freq_data = np.absolute(freq_data)
    freq_data = np.log10(freq_data)

    return freq_data;

def feature_extraction(matdata):
    """
    Transforms example data into x vector
    """
    return fft(matdata)
