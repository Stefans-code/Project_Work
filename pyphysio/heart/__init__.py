# coding=utf-8
# from __future__ import division
import numpy as _np
from ..processing import Algorithm as _Algorithm
from ..signal import create_signal
from ..processing.filters import IIRFilter as _IIRFilter
from ..processing.tools import SignalRange as _SignalRange, Minima as _Minima, Diff as _Diff, PeakDetection as _PeakDetection
import itertools as _itertools

# IBI ESTIMATION
class BeatFromBP(_Algorithm):
    """
    Identify the beats in a Blood Pulse (BP) signal and compute the IBIs.
    Optimized to identify the percussion peak.

    Optional parameters
    -------------------
    
    bpm_max : int, (1, 400], default=120
        Maximal expected heart rate (in beats per minute)
    win_pre : float, (0, 1], default=0.25
        Portion (in seconds) to consider before the candidate beat position where to look for the beat
    win_post : float, (0, 1], default=0.05
        Portion (in seconds) to consider after the candidate beat position where to look for the beat


    Returns
    -------
    ibi : UnevenlySignal
        Inter beat interval values at percussion peaks

    Notes
    -----
    Please cite:
        Bizzego, Andrea, and Cesare Furlanello. "DBD-RCO: Derivative Based Detection And Reverse Combinatorial Optimization To Improve Heart Beat Detection For Wearable Devices." bioRxiv (2017): 118943.
    """
    
    def __init__(self, bpm_max=120, win_pre=None, win_post=None):
        ibi_min = 60/bpm_max
        
        if win_pre is None:
            win_pre = ibi_min / 2
        if win_post is None:
            win_post = ibi_min / 5
        
        assert 0 < win_pre <= ibi_min, "win_pre value should be between 0 and 60/bpm_max"
        assert 0 < win_post <= ibi_min, "win_post peak value should be between 0 and 60/bpm_max"
        
        _Algorithm.__init__(self, bpm_max=bpm_max, win_pre=win_pre, win_post=win_post)
        self.dimensions = {'time':0}

    def algorithm(self, signal):
        
        params = self._params
        fsamp = signal.p.get_sampling_freq()
        bpm_max = params["bpm_max"]
        
        win_pre = params["win_pre"] * fsamp
        win_post = params["win_post"] * fsamp
        
        fmax = bpm_max / 60
        ibi_min = 1 / fmax

        times = signal.p.get_times()

        # STAGE 1 - EXTRACT BEAT POSITION SIGNAL
        # filtering
        signal_f = _IIRFilter(fp=1.2 * fmax, fs=3 * fmax, ftype='ellip')(signal)
        # find range for the adaptive peak detection
        delta = 0.5 * _SignalRange(win_len=1.5 / fmax, win_step=1 / fmax)(signal_f)
        
        delta = delta.values.ravel()

        #adjust for delta values equal to 0
        idx_delta_zeros = _np.where(delta==0)[0]
        idx_delta_nozeros = _np.where(delta>0)[0]
        delta[idx_delta_zeros] = _np.min(delta[idx_delta_nozeros])
        
        # detection of candidate peaks
        maxima = _PeakDetection(delta=delta, refractory=ibi_min, start_max=True, return_peaks=True)(signal_f)
        maxp = _np.where(~_np.isnan(maxima.values))[0].ravel()
        
        if maxp[0] == 0:
            maxp = maxp[1:]

        # STAGE 2 - IDENTIFY PEAKS using the signal derivative
        # compute the signal derivative
        dxdt = _Diff()(signal).values
        
        true_peaks = []
        # for each candidate peak find the correct peak
        for idx_beat in maxp:
            start_ = int(idx_beat - win_pre)
            if start_ < 0:
                start_ = 0

            stop_ = int(idx_beat + win_post)
            if stop_ > len(dxdt):
                stop_ = -1

            # select portion of derivative where to search
            obs = dxdt[start_:stop_]
            peak_obs = _np.argmax(obs)
            i_end = 1
            while peak_obs == (len(obs) - i_end):
                peak_obs = _np.argmax(obs[:-i_end])
                i_end +=1
                
            true_obs = dxdt[start_ + peak_obs: stop_]
            
            true_obs = create_signal(abs(true_obs), 
                                     sampling_freq = fsamp,
                                     start_time = times[start_ + peak_obs])
            
            # find the 'first minimum' (zero) the derivative (peak)
            minima = _Minima(win_len=0.1, win_step=0.025, method='windowing')(true_obs)
                        
            idx_mins = _np.where(~_np.isnan(minima.p.main_signal.values))[0].ravel()

            if len(idx_mins) >= 1:
                peak = idx_mins[0]
                true_peaks.append(start_ + peak_obs + peak + 1)
            # else:
            #     print('Peak not found; idx_beat: ' + str(idx_beat))
            #     pass
        true_peaks = _np.array(true_peaks)
        # STAGE 3 - FINALIZE computing IBI
        t_ibi = true_peaks / fsamp
        v_ibi = _np.diff(t_ibi)
        v_ibi = _np.insert(v_ibi, 0, v_ibi[0])

        ibi_scaffold = _np.nan* _np.zeros(len(signal.values))
        ibi_scaffold[true_peaks] = v_ibi
        
        return ibi_scaffold

class BeatFromECG(_Algorithm):
    """
    Identify the beats in an ECG signal and compute the IBIs.

    Optional parameters
    -------------------
    
    bpm_max : int, (1, 400], default=120
        Maximal expected heart rate (in beats per minute)
    delta : float, >=0, default=0
        Threshold for the peak detection. By default it is computed from the signal (adaptive thresholding)
    k : float, (0,1), default=0.7
        Ratio at which the signal range is multiplied (when delta = 0)

    Returns
    -------
    ibi : UnevenlySignal
        Inter beat interval values at percussion peaks

    Notes
    -----
        This algorithms looks for maxima in the signal which are followed by values lower than a delta value. 
        The adaptive version estimates the delta value adaptively.
    """

    def __init__(self, bpm_max=120, delta=0, k=0.7):
        if not 10 < bpm_max < 400:
            self.warn("Parameter bpm_max out of reasonable range (10, 400)")
        assert delta >= 0, "Delta value should be positive (or equal to 0 if automatically computed)"
        assert 0 < k < 1, "K coefficient must be in the range (0,1)"
        _Algorithm.__init__(self, bpm_max=bpm_max, delta=delta, k=k)
        self.dimensions = {'time':0}

    def algorithm(self, signal):
        params = self._params
        bpm_max, delta, k = params["bpm_max"], params["delta"], params["k"]
        fmax = bpm_max / 60
        
        fsamp = signal.p.get_sampling_freq()
        
        if delta == 0:
            delta = k * _SignalRange(win_len=2 / fmax, win_step=0.5 / fmax, smooth=False)(signal)
            delta = _np.array(delta).ravel()
        
        #adjust for delta values equal to 0
        idx_delta_zeros = _np.where(delta==0)[0]
        idx_delta_nozeros = _np.where(delta>0)[0]
        delta[idx_delta_zeros] = _np.min(delta[idx_delta_nozeros])
        
        refractory = 1 / fmax
        
        #find beats
        maxp = _PeakDetection(delta=delta, refractory=refractory, start_max=True)(signal)
        maxp = _np.array(maxp).ravel()
        
        if maxp[0] == 0:
            maxp = maxp[1:]

        idx_beats = _np.where(~_np.isnan(maxp))[0]
        
        times_beats = idx_beats / fsamp
        
        ibi_values = _np.diff(times_beats)

        ibi_values = _np.insert(ibi_values, 0, ibi_values[0])
        
        ibi_scaffold = _np.nan* _np.zeros(len(signal.values))
        
        ibi_scaffold[idx_beats] = ibi_values

        return ibi_scaffold

class RemoveBeatOutliers(_Algorithm):
    """
    Detects outliers in the IBI signal. 
    
    Optional parameters
    -------------------
    
    cache : int, >0,  default=3
        Number of IBI to be stored in the cache for adaptive computation of the interval of accepted values
    sensitivity : float, >0, default = 0.25
        Relative variation from the current IBI median value of the cache that is accepted
    ibi_median : float, >=0, default = 0
        IBI value use to initialize the cache. By default (ibi_median=0) it is computed as median of the input IBI
    
    Returns
    -------
    id_bad_ibi : numpy.array
        Identifiers of wrong beats
    
    Notes
    -----
    It only detects outliers. You should manually remove outliers using FixIBI
    
    """

    def __init__(self, ibi_median=0, cache=3, sensitivity=0.25):
        assert ibi_median >= 0, "IBI median value should be positive (or equal to 0 for automatic computation"
        assert cache >= 1, "Cache size should be greater than 1"
        assert sensitivity > 0, "Sensitivity value shlud be positive"

        _Algorithm.__init__(self, ibi_median=ibi_median, cache=cache, sensitivity=sensitivity)
        self.dimensions = {'time': 0 }
   
    def algorithm(self, signal):
        params = self._params
        cache, sensitivity, ibi_median = params["cache"], params["sensitivity"], params["ibi_median"]

        ibi_values = signal.p.get_values()
        idx_values = _np.where(~_np.isnan(ibi_values))
        
        ibi_values = ibi_values[idx_values]
        
        if ibi_median == 0:
            ibi_expected = float(_np.median(ibi_values))
        else:
            ibi_expected = float(ibi_median)

        id_good = []
        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        # missings = []
        for i in range(len(ibi_values)):

            curr_median = _np.median(ibi_cache)
            curr_ibi = ibi_values[i]

            if (curr_ibi < curr_median * (1 + sensitivity)) & \
                (curr_ibi > curr_median * (1 - sensitivity)):  # good peak
                id_good.append(i)  # append ibi id to the list of bad ibi
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                counter_bad = 0
            else:
                counter_bad += 1

            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                counter_bad = 0
        
        ibi_scaffold = _np.nan * _np.zeros(len(signal.values))
        
        idx_values_correct = idx_values[0][id_good]
        ibi_values_correct = ibi_values[id_good]
        
        ibi_scaffold[idx_values_correct] = ibi_values_correct
        
        return ibi_scaffold

class BeatOptimizer(_Algorithm):
    """
    Optimize detection of errors in IBI estimation.
    
    Optional parameters
    -------------------

    B : float, >0, default = 0.25
        Ball radius in seconds to allow pairing between forward and backward beats
    cache : int, >0,  default = 3
        Number of IBI to be stored in the cache for adaptive computation of the interval of accepted values
    sensitivity : float, >0, default = 0.25
        Relative variation from the current IBI median value of the cache that is accepted
    ibi_median : float, >=0, default = 0
        IBI value use to initialize the cache. By default (ibi_median=0) it is computed as median of the input IBI

    Returns
    -------
    ibi : UnevenlySignal
        Optimized IBI signal

    Notes
    -----
        Bizzego et al., *DBD-RCO: Derivative Based Detection and Reverse Combinatorial Optimization 
        to improve heart beat detection for wearable devices for info about the algorithm*
    """

    def __init__(self, b=0.25, ibi_median=0, cache=3, sensitivity=0.25):
        assert b > 0, "Ball radius should be positive"
        assert ibi_median >= 0, "IBI median value should be positive (or equal to 0 for automatic computation"
        assert cache >= 1, "Cache size should be greater than 1"
        assert sensitivity > 0, "Sensitivity value shlud be positive"

        _Algorithm.__init__(self, B=b, ibi_median=ibi_median, cache=cache, sensitivity=sensitivity)
        self.dimensions = {'time': 0 }
        
    @classmethod
    def get_signal_type(cls):
        return ['IBI']

    @classmethod
    def algorithm(cls, signal, params):
        b, cache, sensitivity, ibi_median = params["B"], params["cache"], params["sensitivity"], params["ibi_median"]

        idx_ibi = signal.get_indices()
        fsamp = signal.get_sampling_freq()

        if ibi_median == 0:
            ibi_expected = _np.median(_np.diff(idx_ibi))
        else:
            ibi_expected = ibi_median

        idx_st = idx_ibi[0]
        idx_ibi = idx_ibi - idx_st

        ###
        # RUN FORWARD:
        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        idx_1 = [idx_ibi[0]]
        ibi_1 = []

        prev_idx = idx_ibi[0]
        for i in _np.arange(1, len(idx_ibi)):
            curr_median = _np.median(ibi_cache)
            curr_idx = idx_ibi[i]
            curr_ibi = curr_idx - prev_idx

            if curr_ibi > curr_median * (1 + sensitivity):  # abnormal peak:
                prev_idx = curr_idx
                ibi_1.append(_np.nan)
                idx_1.append(curr_idx)
                counter_bad += 1
            elif curr_ibi < curr_median * (1 - sensitivity):  # abnormal peak:
                counter_bad += 1
            else:
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                prev_idx = curr_idx
                ibi_1.append(curr_ibi)
                idx_1.append(curr_idx)

            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                # action_message('Cache re-initialized - ' + str(curr_idx))  # , RuntimeWarning) # message
                counter_bad = 0

        ###
        # RUN BACKWARD:
        idx_ibi_rev = idx_ibi[-1] - idx_ibi
        idx_ibi_rev = idx_ibi_rev[::-1]

        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        idx_2 = [idx_ibi_rev[0]]
        ibi_2 = []

        prev_idx = idx_ibi_rev[0]
        for i in _np.arange(1, len(idx_ibi_rev)):
            curr_median = _np.median(ibi_cache)
            curr_idx = idx_ibi_rev[i]
            curr_ibi = curr_idx - prev_idx

            # print([curr_median*(1+sensitivity), curr_median*(1-sensitivity), curr_median])
            if curr_ibi > curr_median * (1 + sensitivity):  # abnormal peak:
                prev_idx = curr_idx
                ibi_2.append(_np.nan)
                idx_2.append(curr_idx)
                counter_bad += 1

            elif curr_ibi < curr_median * (1 - sensitivity):  # abnormal peak:
                counter_bad += 1
            else:
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                prev_idx = curr_idx
                ibi_2.append(curr_ibi)
                idx_2.append(curr_idx)

            if counter_bad == cache:  # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                # action_message('Cache re-initialized - ' + str(curr_idx))  # , RuntimeWarning) # OK Message
                counter_bad = 0

        idx_2 = -1 * (_np.array(idx_2) - idx_ibi_rev[-1])
        idx_2 = idx_2[::-1]
        ibi_2 = ibi_2[::-1]

        ###
        # add indexes of idx_ibi_2 which are not in idx_ibi_1 but close enough
        b = b * fsamp
        for i_2 in _np.arange(1, len(idx_2)):
            curr_idx_2 = idx_2[i_2]
            if not (curr_idx_2 in idx_1):
                i_1 = _np.where((idx_1 >= curr_idx_2 - b) & (idx_1 <= curr_idx_2 + b))[0]
                if not len(i_1) > 0:
                    idx_1 = _np.r_[idx_1, curr_idx_2]
        idx_1 = _np.sort(idx_1)

        ###
        # create pairs for each beat
        pairs = []
        for i_1 in _np.arange(1, len(idx_1)):
            curr_idx_1 = idx_1[i_1]
            if curr_idx_1 in idx_2:
                pairs.append([curr_idx_1, curr_idx_1])
            else:
                i_2 = _np.where((idx_2 >= curr_idx_1 - b) & (idx_2 <= curr_idx_1 + b))[0]
                if len(i_2) > 0:
                    i_2 = i_2[0]
                    pairs.append([curr_idx_1, idx_2[i_2]])
                else:
                    pairs.append([curr_idx_1, curr_idx_1])
        pairs = _np.array(pairs)

        ########################################
        # define zones where there are different values
        diff_idxs = pairs[:, 0] - pairs[:, 1]
        diff_idxs[diff_idxs != 0] = 1
        diff_idxs = _np.diff(diff_idxs)

        starts = _np.where(diff_idxs > 0)[0]
        stops = _np.where(diff_idxs < 0)[0]
        
        if len(starts)==0: # no differences
            return signal
        
        if len(stops)==0:
            stops = _np.array([starts[-1] + 1])
            
        if starts[0] >= stops[0]:
            stops = stops[1:]

        stops += 1

        if len(starts) > len(stops):
            stops = _np.r_[stops, starts[-1] + 1]

        # split long sequences
        new_starts = _np.copy(starts)
        new_stops = _np.copy(stops)

        add_index = 0
        lens = stops - starts
        for i in _np.arange(len(starts)):
            l = lens[i]
            if l > 10:
                curr_st = starts[i]
                curr_sp = stops[i]
                new_st = _np.arange(curr_st, curr_sp, 4)
                new_sp = new_st + 4
                new_sp[-1] = curr_sp
                new_starts = _np.delete(new_starts, i + add_index)
                new_stops = _np.delete(new_stops, i + add_index)
                new_starts = _np.insert(new_starts, i + add_index, new_st)
                new_stops = _np.insert(new_stops, i + add_index, new_sp)
                add_index = add_index + len(new_st) - 1

        starts = new_starts
        stops = new_stops

        ########################################
        # find best combination
        idx_out = _np.copy(pairs[:, 0])
        for i in _np.arange(len(starts)):
            i_st = starts[i]
            i_sp = stops[i]

            if i_sp > len(idx_out) - 1:
                i_sp = len(idx_out) - 1

            curr_portion = _np.copy(pairs[i_st - 1: i_sp + 1, :])

            best_portion = None
            best_error = _np.Inf

            combinations = list(_itertools.product([0, 1], repeat=i_sp - i_st - 1))
            for comb in combinations:
                cand_portion = _np.copy(curr_portion[:, 0])
                for k in range(len(comb)):
                    bit = comb[k]
                    cand_portion[k + 2] = curr_portion[k + 2, bit]
                cand_error = sum(abs(_np.diff(_np.diff(cand_portion))))
                if cand_error < best_error:
                    best_portion = cand_portion
                    best_error = cand_error
            idx_out[i_st - 1: i_sp + 1] = best_portion

        ###
        # finalize arrays
        idx_out = _np.array(idx_out) + idx_st
        ibi_out = _np.diff(idx_out)
        ibi_out = _np.r_[ibi_out[0], ibi_out]


        return _UnevenlySignal(ibi_out, sampling_freq=signal.get_sampling_freq(), signal_type="IBI",
                               start_time=signal.get_start_time(), x_values=idx_out, x_type='indices',
                               duration=signal.get_duration())