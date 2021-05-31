import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyphysio as ph

from pyphysio import TestData
bvp_data = TestData.bvp()

fsamp = 2048
bvp = ph.EvenlySignal(values = bvp_data, sampling_freq = fsamp)
ibi = ph.BeatFromBP()(bvp)

#%%
np.median(ibi)

#%%
ibi_corrected = ph.Annotate(bvp, ibi).ibi_ok

#%%
id_wrong_ibi = ph.BeatOutliers()(ibi_corrected)


# In[13]:


id_wrong_ibi


# In[12]:


ibi_no_outliers = ph.FixIBI(id_wrong_ibi)(ibi_corrected)


# In[ ]:


ibi_no_outliers.plot('.-')


# ## Step 3: Physiological Indicators

# In[ ]:


# check label
ax1 = plt.subplot(211)
ibi_no_outliers.plot('.-')

plt.subplot(212, sharex = ax1)
label.plot('.-')
plt.show()


# In[ ]:


# define a list of indicators we want to compute
hrv_indicators = [ph.Mean(name='RRmean'), ph.StDev(name='RRstd'), ph.RMSSD(name='rmsSD')]


# In[ ]:


#fixed length windowing
fixed_length = ph.FixedSegments(step = 5, width = 10, labels = label)

indicators, col_names = ph.fmap(fixed_length, hrv_indicators, ibi)


# In[ ]:


results = pd.DataFrame(indicators, columns = col_names)
display(results)


# In[ ]:




