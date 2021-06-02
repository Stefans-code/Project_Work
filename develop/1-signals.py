#!/usr/bin/env python
# coding: utf-8

# # pyphysio tutorial
# 
# ## 1. Signals
# 
# The class `Signal` together with the class `Algorithm` are the two main classes in pyphysio.
# 
# In this first tutorial we will see how the class `Signal` can be used to facilitate the management and processing of signals.
# 
# A signal is an ordered vector of timestamp-value pairs, where the timestamp is the instant at which the measured phenomenon had that value.
# In pyphysio a signal is represented by the class **Signal** which extends the numpy.ndarray class.
# 
# In this part we will see the different types of signals that can be defined and their properties.
# 
# We start importing the packages needed for the tutorial:

# In[1]:


# import packages
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# And then we import two classess of `pyphysio`: `EvenlySignal` and `UnevenlySignal`, both subclassess of the abstract class `Signal':

# In[2]:


# import the Signal classes
from pyphysio import Signal


# ### 1.1 EvenlySignal
# 
# When the signal is sampled with a fixed sampling frequency it is sufficient to know the timestamp at which the acquisition started and the sampling frequency (assumed to be constant) to reconstruct the timestamp of each sample. This type of signal is represented by the class **EvenlySignal**.
# 
# Therefore to create an instance of **EvenlySignal** these are the input attributes needed:
# * ``values`` : (unidimensional numpy array) values of the signal;
# * ``sampling_freq`` : (float>0) sampling frequency;
# * ``start_time`` : (float) temporal reference of the start of the signal. This is optional, if omitted it will set to 0;
# * ``info`` : (dict) dictionary that will contain general (custom) information about the signal.
# 
# Class functions are provided to facilitate the management and processing of signals. For instance:
# * `get_...()` and `set_...()` type functions can be used to check/set signal attributes;
# * `plot()` will plot the signal using matplotlib;
# * `segment_time(t_start, t_stop)` and `segment_idx(idx_start, idx_stop)` can be used to extract a portion of the signal;
# * `resample(fout)` can be used to change the sampling frequency.
# 
# **TODO: meaning of axes and slicing**

# #### 1.1.1 Creation of a `EvenlySignal` object
# 
# In the following we generate a fake `EvenlySignal` using random generated numbers. Then we will use the methods provided by the class to inspect the signal characteristics:

# In[3]:


# create a signal

## create fake data
np.random.seed(4)
signal_values = np.random.uniform(0, 1, size = 1000)

## set the sampling frequency
fsamp = 100 # Hz

## set the starting time
tstart = 100 # s

## create the Evenly signal
s_fake = Signal(values = signal_values, sampling_freq = fsamp, start_time = tstart, info  = {'type': 'fake', 'subject': 'SUB001'})

# chech signal properties
print('Sampling frequency: {}'.format( s_fake.get_sampling_freq() ))
print('Start time:         {}'.format( s_fake.get_start_time() ))
print('End time:           {}'.format( s_fake.get_end_time() ))
print('Duration:           {}'.format( s_fake.get_duration() ))
print('Signal info  :      {}'.format( s_fake.get_info() ))
print('First ten instants: {}'.format( s_fake.get_times()[0:10] ))


# #### 1.1.2 Plotting a signal
# 
# A shortcut is provided in `pyphysio` to plot a signal, using the `matplotlib` library:

# In[4]:


## plot
s_fake.plot()


# #### 1.3 Working with physiological signals
# 
# In this second example we import the sample data included in `pyphysio` to show how the `EvenlySignal` class can be used to represent physiological signals:

# In[5]:


import pyphysio as ph


# In[6]:


# import data from included examples
from pyphysio import TestData

ecg_data = TestData.ecg()
eda_data = TestData.eda()


# The imported values can be used to create two new signals of the `EvenlySignal` class.
# 
# Note that we set different starting times for the ecg and the eda signal:

# In[7]:


# create two signals
fsamp = 2048
tstart_ecg = 15
tstart_eda = 5

ecg = Signal(values = ecg_data, 
             sampling_freq = fsamp, 
             start_time = tstart_ecg)

eda = Signal(values = eda_data, 
             sampling_freq = fsamp, 
             start_time = tstart_eda)


# In the following plot note that the EDA signal start 10 seconds before the ECG signal.
# 
# Using the `start_time` parameter is therefore possible to manually synchronize multiple signals.

# In[8]:


# plot
ax1 = plt.subplot(211)
ecg.plot()
plt.subplot(212, sharex=ax1)
eda.plot()


# In[9]:


# check signal properties
print('ECG')
print('Sampling frequency: {}'.format( ecg.get_sampling_freq() ))
print('Start time:         {}'.format( ecg.get_start_time() ))
print('End time:           {}'.format( ecg.get_end_time() ))
print('Duration:           {}'.format( ecg.get_duration() ))
print('First ten instants: {}'.format( ecg.get_times()[0:10] ))
print('')
print('EDA')
print('Sampling frequency: {}'.format( eda.get_sampling_freq() ))
print('Start time:         {}'.format( eda.get_start_time() ))
print('End time:           {}'.format( eda.get_end_time() ))
print('Duration:           {}'.format( eda.get_duration() ))
print('First ten instants: {}'.format( eda.get_times()[0:10] ))


# #### 1.4 Managing the sampling frequency
# The sampling frequency of a signal is defined before the acquisition. However it is possible to numerically change it in order to oversample or downsample the signal, according to the signal type and characteristics.
# 
# Note in the plot below the effect of downsampling the ECG.

# In[10]:


# resampling
ecg_128 = ecg.resample(fout=128)

ecg.plot() # plotting the original signal
ecg_128.plot('.') # plotting the samples of the downsampled signal
plt.xlim((40,42)) # setting the range of the x axis between 40 and 42 seconds


# #### 1.5 Multi-channels, multi-components
# 
# _TODO_
# 
# 

# In[12]:


fake_multi_values = np.random.normal(size = (1000, 3, 5))

fake_multi_values = fake_multi_values * np.array([1,2,3]).reshape(1,-1,1)
fake_multi_values = fake_multi_values + np.array([10,20,30,40,50]).reshape(1,1,-1)


fake_multi_signal = ph.Signal(fake_multi_values, sampling_freq = 10)

fake_multi_signal.plot()


# ### 1.2 UnevenlySignal
# 
# Other types of signals, for instance triggers indicating occurrences of heartbeats or events, are series of samples which are not equally temporally spaced. Thus the sampling frequency is not fixed and it is necessary to store the timestamp of each sample. This type of signals is represented by the class **UnevenlySignal**.
# 
# Therefore to create an instance of **UnevenlySignal** additional input attributes are needed:
# * `x_values` : (unidimensional numpy array) information about the temporal position of each sample. Should be of the same size of ``values``;
# * `x_type` : (`'instants'` or `'indices'`) indicate what type of x_values have been used.
# 
# Two ways are allowed to define an **UnevenlySignal**:
# 1. by defining the indexes (`x_type='indices'`): x_values are indices of an array and the instants are automatically computed using the information from the `sampling_frequency` and the `start_time`. 
# 2. by defining the instants (`x_type='instants'`): x_values are instants and the indices are automatically computed using the information from the `sampling_frequency`. The `start_time` (should not be passed) is set to the first value of the `x_values`. 
# 
# As a general rule, the `start_time` is always associated to the index 0.
# 
# An additional class function is provided to transform an **UnevenlySignal** to an **EvenlySignal**:
# * `to_evenly()` create an `EvenlySignal` by interpolating the signal with given signal sampling frequency.

# #### 1.2.1 Creating an `UnevenlySignal` object
# 
# In the following we generate a fake `UnevenlySignal` using random generated numbers. 
# We will use two methods to provide the temporal information about each sample:
# 1. by providing the information about the indices;
# 2. by providing the informatino about the instants.
# 
# Then we will use the provided methods to inspect the signal characteristics:

# In[14]:


## create fake data
signal_values = np.arange(100)


# Create an `UnevenlySignal` object providing the indices:

# In[15]:


## create fake indices
idx = np.arange(100)
idx[-1] = 125

## set the sampling frequency
fsamp = 10 # Hz

## set the starting time
tstart = 10 # s

## create an Unevenly signal defining the indices
x_values_idx = idx


s_fake_idx = Signal(values = signal_values, 
                    sampling_freq = fsamp, 
                    start_time = tstart,
                    x_values = x_values_idx, 
                    x_type = 'indices')


# Create an `UnevenlySignal` object providing the instants:

# In[19]:
import numpy as np
from pyphysio import Signal

## create an Unevenly signal defining the instants

x_values_time = np.arange(100)/fsamp
x_values_time[-1] = 12.5
x_values_time += 10


## set the starting time
tstart = 0 

s_fake_time = Signal(values = signal_values, 
                     sampling_freq = fsamp, 
                     start_time = tstart,
                     x_values = x_values_time, 
                     x_type = 'instants')

s_fake_time_evenly = s_fake_time.fill(kind = 'linear')

# Note in the following plot that the interval between the last two samples is different from all the others:

# In[20]:


#plot
ax1=plt.subplot(211)
s_fake_idx.plot('.')
plt.subplot(212, sharex=ax1)
s_fake_time.plot('.')


# In[21]:


# note that the times are the same but not the starting_time nor the indices:

# check samples instants
print('Instants:')
print(s_fake_idx.get_times())
print(s_fake_time.get_times())

# check samples indices
print('Indices:')
print(s_fake_idx.get_indices())
print(s_fake_time.get_indices())

# check start_time
print('Start time:')
print(s_fake_idx.get_start_time())
print(s_fake_time.get_start_time())


# In[22]:


# chech signal properties
print('Defined by Indices')
print('Sampling frequency: {}'.format( s_fake_idx.get_sampling_freq() ))
print('Start time:         {}'.format( s_fake_idx.get_start_time() ))
print('End time:           {}'.format( s_fake_idx.get_end_time() ))
print('Duration:           {}'.format( s_fake_idx.get_duration() ))
print('First ten instants: {}'.format( s_fake_idx.get_times()[0:10] ))
print('')
print('Defined by Instants')
print('Sampling frequency: {}'.format( s_fake_time.get_sampling_freq() ))
print('Start time:         {}'.format( s_fake_time.get_start_time() ))
print('End time:           {}'.format( s_fake_time.get_end_time() ))
print('Duration:           {}'.format( s_fake_time.get_duration() ))
print('First ten instants: {}'.format( s_fake_time.get_times()[0:10] ))


# #### 1.2.2 From `UnevenlySignal` to `EvenlySignal`

# It is possible to obtain an `EvenlySignal` from an `UnevenlySignal` by interpolation, using the method `to_evenly` of the class `UnevenlySignal`:

# In[23]:


# to_evenly
s_fake_time_evenly = s_fake_time.fill(kind = 'linear')


# Note how the interval between the last two samples has been interpolated in the `EvenlySignal` version (blue) of the original signal (yellow):

# In[19]:


s_fake_time_evenly.plot('.b')
s_fake_time.plot('.y')

# check type
print(type(s_fake_time_evenly))
print(type(s_fake_time))


# ### 1.3 Segmentation of signals
# 
# **TODO: update**
# 
# Two general class functions are provided to segment a signal:
# 1. `segment_time(t_start, t_stop)` is used to extract a portion of the signal between the instants `t_start` and
# `t_stop`;
# 2. `segment_idx(idx_start, idx_stop)` is used to extract a portion of the signal between the indices `idx_start` and `idx_stop`.
# 
# The output signal will inherit **`sampling_freq`** and **`signal_nature`** but the **`start_time`** will be set to **`t_start`** or to the instant corresponding to **`idx_start`** accordingly to the method used.

# In[20]:


# segmentation of ES
ecg_segment = ecg.segment_time(45, 54)
eda_segment = eda.segment_time(45, 54)


# In[21]:


# plot
ax1 = plt.subplot(211)
ecg.plot()
ecg_segment.plot('r')

plt.subplot(212, sharex=ax1)
eda.plot()
eda_segment.plot('r')

print(ecg_segment.get_start_time())


# In[22]:


# segmentation of US

s_fake_idx_segment = s_fake_idx.segment_time(10.5, 18)
s_fake_time_segment = s_fake_time.segment_time(10.5, 18)


# In[23]:


# plot
ax1 = plt.subplot(211)
s_fake_idx.plot('.')
s_fake_idx_segment.plot('.r')

plt.subplot(212, sharex=ax1)
s_fake_time.plot('.')
s_fake_time_segment.plot('.r')

print(s_fake_time_segment.get_start_time())


# In[ ]:




