"""
Plots time series data
"""

import os
import numpy as np
from matplotlib import pyplot as plt

def plot_time_series(matdata, filename='unnamed.png', plot_dir='plots/time/'):

    # Sampling Rate
    fs = matdata["dataStruct"]['iEEGsamplingRate'][0][0][0][0]
    # Total Samples
    n_samples = matdata["dataStruct"]["nSamplesSegment"][0][0][0][0]
    # [n_examples, n_electrodes]
    time_data = matdata['dataStruct']['data'][0][0]

    # Sample times
    t = np.linspace(0, (1/fs * n_samples), n_samples, endpoint=False)

    plt.plot(t, time_data)
    plt.xlabel('Time (s)')
    plt.ylabel('Electrode Voltage Diff')

    plot_path = os.path.dirname(os.path.realpath(__file__)) + '/' + plot_dir + '/'
    plt.savefig(plot_path + filename)

def plot_fft(freq_data, bandwidth, filename='unnamed.png', plot_dir='plots/fft/'):

    # Sample frequencies
    f = np.linspace(0, bandwidth-1, bandwidth, endpoint=False)

    plt.plot(f, freq_data)
    plt.title(filename)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (10^y)')

    plot_path = os.path.dirname(os.path.realpath(__file__)) + '/' + plot_dir + '/'
    plt.savefig(plot_path + filename)
