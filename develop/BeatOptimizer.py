import pyphysio as ph
import pyphysio.specialized.heart as heart
import pyphysio.filters as filters
import pyphysio.utils as utils
import matplotlib.pyplot as plt
import itertools as _itertools

import os
import numpy as _np
import xarray as _xr

datadir = '/home/bizzego/UniTn/data/DBD/pyphysio/ppg-dalia'

SESSION = 'sitting'
SUBJECT = 'S7'
bvp = _xr.load_dataset(f'{datadir}/{SESSION}/bvp/{SUBJECT}')
bvp.attrs['history'] = [bvp.attrs['history']]

# bvp = bvp.p.segment_time(4400,4600)

#%%
#compute average cardiac frequency
bvp_ = filters.FIRFilter(fp=[0.5, 3.5], fs=[0.4, 4])(bvp)

psd = utils.PSD(method = 'fft')(bvp_)
psd = psd.p.main_signal
av_freq = psd.freq.values[_np.argmax(psd.values.ravel())]
bpm_max = 60*av_freq*1.5

#%% detect ibi   
ibi = heart.BeatFromBP(bpm_max = bpm_max)(bvp)

#%%
ibi_rco = heart.BeatOptimizer()(ibi, bvp)
ibi_rco_corr = heart.RemoveBeatOutliers()(ibi_rco)

#%
ibi_ = ibi.p.process_na('remove')
ibi_rco_ = ibi_rco.p.process_na('remove')
ibi_rco_corr_ = ibi_rco_corr.p.process_na('remove')

#%
ibi_.p.plot()
ibi_rco_.p.plot()
ibi_rco_corr_.p.plot()

#%%
signal = ibi.p.main_signal

def _add_peaks(t_prev, t_curr, ibi_cache, bvp_signal=None):
    duration_interval = t_curr - t_prev
    ibi_median = _np.median(ibi_cache)
    n_expected_beats = _np.round(duration_interval / ibi_median)-1
    t_targets = _np.linspace(t_prev, t_curr, 2+int(n_expected_beats))[1:-1]
    
    if bvp_signal is None:
        return t_targets

    if _np.std(ibi_cache) != 0:
        ibi_min = 0.9*_np.min(ibi_cache)
        ibi_max = 1.1*_np.min(ibi_cache)
    else:
        ibi_min = (1 - sensitivity)*ibi_median
        ibi_max = (1 + sensitivity)*ibi_median
    
    t_pre = ibi_median - ibi_min
    t_post = ibi_max - ibi_median
    
    t_targets_new = []
    for t in t_targets:
        bvp_portion = bvp_signal.p.segment_time(t-t_pre, t+t_post)
        bvp_values = bvp_portion.p.main_signal.values.ravel()
        bvp_times = bvp_portion.p.get_times()
        
        #search local max using derivative
        dbvp = _np.diff(bvp_values)
        
        #sort bvp values from bigger to smaller - focus on 5 biggest values
        idx_bvp_sorted = _np.argsort(bvp_values)[::-1][:5]
        #sort dbvp values from smaller (~0 = local max or min) - focus on 5 biggest values
        idx_dbvp_sorted = _np.argsort(dbvp)[:5]
        
        #find idx which has the highest value and lowest dbvp
        idx_coincident = _np.argmin(abs(idx_bvp_sorted - idx_dbvp_sorted))
        idx_max = idx_bvp_sorted[idx_coincident] -1
        t_targets_new.append(bvp_times[idx_max])

    return t_targets_new


assert signal.p.get_sampling_freq() != 'unevenly', "This algorithm should be applied to evenly IBI. Avoid processing nans before"

# params = self._params
# cache, sensitivity, ibi_median = params["cache"], \
#     params["sensitivity"], params["ibi_median"]
cache = 3
sensitivity = 0.25
ibi_median = 0
    
fsamp = signal.p.get_sampling_freq()

ibi_values = signal.p.get_values().ravel()
idx_values = _np.where(~_np.isnan(ibi_values))

ibi_values = ibi_values[idx_values]
t_ibi = signal.p.get_times()[idx_values]

if ibi_median == 0:
    ibi_expected = _np.median(ibi_values)
else:
    ibi_expected = ibi_median

#%
# RUN FORWARD:
ibi_cache = _np.repeat(ibi_expected, cache)
counter_bad = 0

# idx_1 = [idx_ibi[0]]
prev_t_ibi = t_ibi[0]
t_ibi_1 = [prev_t_ibi]

for id_ibi in _np.arange(1, len(t_ibi)):
    #%
    curr_t_ibi = t_ibi[id_ibi]
    curr_median = _np.median(ibi_cache)
    curr_ibi = curr_t_ibi - prev_t_ibi
    
    if curr_ibi > curr_median * (1 + sensitivity):  
        # abnormal peak: probably a missing beat
        counter_bad += 1
        
        #we assume there are missing beat(s) in between
        t_missed_peaks = _add_peaks(prev_t_ibi, curr_t_ibi, ibi_cache, bvp.p.main_signal)
        for t in t_missed_peaks:
            t_ibi_1.append(t) 
        
        t_ibi_1.append(curr_t_ibi)
        prev_t_ibi = curr_t_ibi
        # print('added', t_half, prev_t_ibi, curr_t_ibi, curr_ibi)
        
    elif curr_ibi < curr_median * (1 - sensitivity):
        # abnormal peak: probably a false beat
        counter_bad += 1
        # print('small', prev_t_ibi, curr_t_ibi, curr_ibi)
    else:
        ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
        t_ibi_1.append(curr_t_ibi)
        prev_t_ibi = curr_t_ibi
        # print('good', prev_t_ibi, curr_t_ibi, curr_ibi)
    
    if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
        ibi_cache = _np.repeat(ibi_expected, cache)
        # action_message('Cache re-initialized - ' + str(curr_idx))  # , RuntimeWarning) # message
        counter_bad = 0

# RUN BACKWARD:
prev_t_ibi = t_ibi[-1]
t_ibi_2 = [prev_t_ibi]

for id_ibi in _np.arange(len(t_ibi)-2,-1,-1):
    #%
    curr_t_ibi = t_ibi[id_ibi]
    curr_median = _np.median(ibi_cache)
    
    curr_ibi = abs(curr_t_ibi - prev_t_ibi)
    
    if curr_ibi > curr_median * (1 + sensitivity): 
        # abnormal peak: probably a missing beat
        counter_bad += 1
        
        #we assume there are missing beat(s) in between
        #note: we change the order of curr_ibi and prev_ibi 
        #as we are going backward
        
        t_missed_peaks = _add_peaks(curr_t_ibi, prev_t_ibi, ibi_cache, bvp.p.main_signal)
        for t in t_missed_peaks:
            t_ibi_2.append(t) 
        t_ibi_2.append(curr_t_ibi)
    
        # print('added', t_half, prev_t_ibi, curr_t_ibi, curr_ibi)
    
        prev_t_ibi = curr_t_ibi
        
        
    elif curr_ibi < curr_median * (1 - sensitivity):
        # abnormal peak: probably a false beat
        # print('small', prev_t_ibi, curr_t_ibi, curr_ibi)
        counter_bad += 1
    
    else:
        # print('good', prev_t_ibi, curr_t_ibi, curr_ibi)
    
        ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
        t_ibi_2.append(curr_t_ibi)
        prev_t_ibi = curr_t_ibi
        
    if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0


t_ibi_1 = _np.array(t_ibi_1)
t_ibi_2 = _np.array(t_ibi_2)[::-1]

###
# create pairs for each beat
pairs = []
for t_1 in t_ibi_1:
    if t_1 in t_ibi_2:
        pairs.append([t_1, t_1])
    else:
        t_2 = t_ibi_2[_np.argmin(abs(t_ibi_2 - t_1))]
        pairs.append([t_1, t_2])

for t_2 in t_ibi_2:
    if t_2 in t_ibi_1:
        new_item = [t_2, t_2]
    else:
        t_1 = t_ibi_1[_np.argmin(abs(t_ibi_1 - t_2))]
        new_item = [t_1, t_2]
    if new_item not in pairs:
        pairs.append(new_item)

pairs = _np.array(pairs)
pairs = _np.sort(pairs, 0)

########################################
# define zones where there are different values
diff_idxs = pairs[:, 0] - pairs[:, 1]
diff_idxs[diff_idxs != 0] = 1
diff_idxs = _np.diff(diff_idxs)

starts = _np.where(diff_idxs == 1)[0]+1
stops = _np.where(diff_idxs == -1)[0]

if len(starts)==0: # no differences
    t_out = t_ibi_1

else:
    if len(stops)==0:
        stops = _np.array([starts[-1] + 1])
        
    if starts[0] >= stops[0]:
        stops = stops[1:]
    
    stops += 1
    
    if len(starts) > len(stops):
        stops = _np.r_[stops, starts[-1] + 1]
    
    assert sum((stops-starts)<1) == 0
    
    #TODO: fix maximum number of combinations 
    # lens = stops - starts
    
    ########################################
    # find best combination
    t_out = _np.copy(pairs[:, 0])

    for i in _np.arange(len(starts)):
        i_st = starts[i]
        i_sp = stops[i]
       

        if i_sp > len(t_out) - 1:
            i_sp = len(t_out) - 1
           
        if (i_sp - i_st) <= 10:
            i_st_ = [i_st]
            i_sp_ = [i_sp]
        
        else:
            print(i_sp-i_st)
            n_ = int(_np.round((i_sp - i_st) / 10))
            idx_cuts = _np.linspace(i_st, i_sp, n_+1).astype(int)
            
            i_st_ = idx_cuts[:-1]
            i_sp_ = idx_cuts[1:]
            
        for i_st, i_sp in zip(i_st_, i_sp_):
            curr_portion = _np.copy(pairs[i_st - 1: i_sp + 1, :])
        
            best_portion = None
            best_error = _np.Inf
        
            combinations = list(_itertools.product([0, 1], repeat=i_sp - i_st))
            for comb in combinations:
                cand_portion = _np.copy(curr_portion[:, 0])
                for i_bit, bit in enumerate(comb):
                    cand_portion[i_bit + 1] = curr_portion[i_bit + 1, bit]
                cand_portion = _np.unique(cand_portion)
                cand_portion_ibi = _np.diff(cand_portion)
                cand_error = _np.std(cand_portion_ibi)
                if cand_error < best_error:
                    best_portion = cand_portion
                    best_error = cand_error
            t_out_replace = _np.nan*_np.zeros(len(curr_portion))
            t_out_replace[0:len(best_portion)] = best_portion
            t_out[i_st - 1: i_sp + 1] = t_out_replace
    
    t_out = t_out[_np.where(~_np.isnan(t_out))[0]]
    t_out = _np.unique(t_out)

v_ibi = _np.diff(t_out)
v_ibi = _np.insert(v_ibi, 0, v_ibi[0])

ibi_scaffold = _np.nan * _np.zeros(len(signal))

idx_ibi = _np.round((t_out - signal.p.get_start_time()) * fsamp).astype(int)
ibi_scaffold[idx_ibi] = v_ibi

#%%
ibi_ = ibi.p.process_na('remove')

bvp.p.plot()
plt.vlines(ibi_.p.get_times(), -100, 100, 'gray')
plt.vlines(t_out, -100, 0, 'r')

#%%
ibi_.p.plot()
plt.plot(t_out, v_ibi, '.')

#%%
ibi_rco_corr = heart.RemoveBeatOutliers()(ibi_rco)

#%%
ibi_ = ibi.p.process_na('remove')
ibi_rco_ = ibi_rco.p.process_na('remove')
ibi_rco_corr_ = ibi_rco_corr.p.process_na('remove')

#%%
ibi_.p.plot()
ibi_rco_.p.plot()
ibi_rco_corr_.p.plot()

