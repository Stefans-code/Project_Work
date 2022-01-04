# import packages
import numpy as np
import matplotlib.pyplot as plt

# import data from included examples
from pyphysio import TestData
from pyphysio import create_signal

ecg_data = TestData.ecg()
eda_data = TestData.eda()

# create two signals
fsamp = 2048
tstart_ecg = 15
tstart_eda = 5

ecg = create_signal(data = ecg_data, sampling_freq = fsamp, start_time = tstart_ecg)

eda = create_signal(data = eda_data, sampling_freq = fsamp, start_time = tstart_eda)

