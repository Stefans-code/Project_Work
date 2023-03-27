import pywt
import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(size=1000)

cA, cB = pywt.wavedec(data, 'db2')
