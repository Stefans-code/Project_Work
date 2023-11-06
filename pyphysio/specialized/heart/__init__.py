# coding=utf-8
# from __future__ import division
import numpy as _np
from scipy.signal import detrend as _detrend
from ..._base_algorithm import _Algorithm
from ...signal import create_signal
from ...filters import IIRFilter as _IIRFilter
from ...utils import SignalRange as _SignalRange, Minima as _Minima, Diff as _Diff, PeakDetection as _PeakDetection
import itertools as _itertools

from ._presets import *
# IBI ESTIMATION

class BeatMSPTD(_Algorithm):
    """
    Identify the beats in a Blood Pulse (BP) signal and compute the IBIs.
    Optimized to identify the percussion peak.
    #code from: https://github.com/peterhcharlton/ppg-beats/blob/main/source/msptd_beat_detector.m
    #paper: Multi-Scale Peak & Trough Detection (Bishop and Ercole 2018)
    #found in paper: Detecting beats in the photoplethysmogram: benchmarking open-source algorithms

    Parameters
    ----------
    win_lem : default 6

    Returns
    -------
    ibi : UnevenlySignal
        Inter beat interval values at percussion peaks

    Notes
    -----
    Please cite:
        Bizzego, Andrea, and Cesare Furlanello. "DBD-RCO: Derivative Based Detection And Reverse Combinatorial Optimization To Improve Heart Beat Detection For Wearable Devices." bioRxiv (2017): 118943.
    """

    def __init__(self, win_len=6, overlap=0.2, tol=0.05):
        # TODO: tol depending on bpm_max?
        _Algorithm.__init__(self, win_len=win_len, overlap=overlap, tol=tol)
        self.dimensions = {'time': 0}

    def algorithm(self, signal):
        params = self._params
        win_len = params["win_len"]
        overlap = params["overlap"]
        tol = params["tol"]

        fsamp = signal.p.get_sampling_freq()
        tol = int(_np.ceil(fsamp*tol))

        no_samps_in_win = int(win_len * fsamp)

        signal_values = signal.values.ravel()

        if len(signal_values) <= no_samps_in_win:
            win_starts = _np.array([0])
        else:
            win_offset = round(no_samps_in_win * (1-overlap))
            win_starts = _np.arange(
                0, len(signal_values)-no_samps_in_win, win_offset).astype(int)

            if win_starts[-1] + no_samps_in_win < len(signal_values):
                win_starts = _np.insert(win_starts, len(
                    win_starts), len(signal_values) - no_samps_in_win)

        peaks = []
        onsets = []

        for i_win, idx_st in enumerate(win_starts):
            # % - extract this window's data
            win_sig = signal_values[idx_st:idx_st+no_samps_in_win+1]

            # TODO: downsampling here?

            # % detect peaks and onsets =========================
            N = len(win_sig)  # % length of signal
            L = int(_np.ceil(N/2)-1)  # ; % max window length

            # Step 1: calculate local maxima and local minima scalograms

            # - detrend
            # % this removes the best-fit straight line
            win_sig_det = _detrend(win_sig)

            # - initialise LMS matrices
            m_max = _np.zeros((L, N))
            m_min = _np.zeros((L, N))

            # - populate LMS matrices
            for k in _np.arange(1, L+1):  # % scalogram scales

                for i in _np.arange(k, N-k):
                    if win_sig_det[i] > win_sig_det[i-k] and win_sig_det[i] > win_sig_det[i+k]:
                        m_max[k-1, i] = 1
                    if win_sig_det[i] < win_sig_det[i-k] and win_sig_det[i] < win_sig_det[i+k]:
                        m_min[k-1, i] = 1

            # Step 2: find the scale with the most local maxima (or local minima)
            # - row-wise summation
            gamma_max = _np.sum(m_max, 1)
            gamma_min = _np.sum(m_min, 1)

            # - find scale with the most local maxima (or local minima)
            idx_lambda_max = _np.argmax(gamma_max)
            idx_lambda_min = _np.argmax(gamma_min)

            # % Step 3: Use lambda to remove all elements of m for which k>lambda
            m_max = m_max[:idx_lambda_max+1, :]
            m_min = m_min[:idx_lambda_min+1, :]

            # Step 4: Find peaks
            # - column-wise summation
            m_max_sum = _np.sum(abs(m_max-1), axis=0)
            m_min_sum = _np.sum(abs(m_min-1), axis=0)
            p = _np.where(m_max_sum == 0)[0]
            t = _np.where(m_min_sum == 0)[0]

            # TODO: downsampling here?

            # % - correct peak indices by finding highest point within tolerance either side of detected peaks

            for i_p, curr_peak in enumerate(p):
                tol_start = curr_peak - tol
                tol_end = curr_peak + tol
                idx_max = _np.argmax(win_sig[tol_start:tol_end+1])
                p[i_p] = curr_peak - tol + idx_max

            # % - correct onset indices by finding highest point within tolerance either side of detected onsets
            for i_o, curr_onset in enumerate(t):
                tol_start = curr_onset - tol
                tol_end = curr_onset + tol
                idx_min = _np.argmin(win_sig[tol_start:tol_end+1])
                t[i_o] = curr_onset - tol + idx_min

            # % - store peaks and onsets
            win_peaks = p + idx_st
            peaks = peaks + list(win_peaks)
            win_onsets = t + idx_st
            onsets = onsets + list(win_onsets)

        # % tidy up detected peaks and onsets (by ordering them and only retaining unique ones)
        peaks = _np.unique(peaks)
        onsets = _np.unique(onsets)

        # STAGE 3 - FINALIZE computing IBI
        t_ibi = peaks / fsamp
        v_ibi = _np.diff(t_ibi)
        v_ibi = _np.insert(v_ibi, 0, v_ibi[0])

        ibi_scaffold = _np.nan * _np.zeros(len(signal_values))
        ibi_scaffold[peaks] = v_ibi

        return ibi_scaffold

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

        _Algorithm.__init__(self, bpm_max=bpm_max,
                            win_pre=win_pre, win_post=win_post)
        self.dimensions = {'time': 0}

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
        signal_f =  _IIRFilter(fp=[0.5*fmax, 1.5*fmax], btype='bandpass')(signal)
        
        # find range for the adaptive peak detection
        delta = 0.5 * _SignalRange(win_len=1.5 / fmax, win_step=1 / fmax)(signal_f)
        delta = delta.values.ravel()

        # adjust for delta values equal to 0
        idx_delta_zeros = _np.where(delta == 0)[0]
        idx_delta_nozeros = _np.where(delta > 0)[0]
        delta[idx_delta_zeros] = _np.min(delta[idx_delta_nozeros])

        # detection of candidate peaks
        maxima = _PeakDetection(
            delta=delta, refractory=ibi_min, return_peaks=True)(signal_f)
        maxp = _np.where(~_np.isnan(maxima.values))[0].ravel()
        
        if len(maxp) == 0:
            return(_np.nan * _np.zeros(len(signal.p.main_signal.values)))
        
        if maxp[0] == 0:
            maxp = maxp[1:]

        # STAGE 2 - IDENTIFY PEAKS using the signal derivative
        # compute the signal derivative
        dxdt = _Diff()(signal).values
        
        import matplotlib.pyplot as plt
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
            
            while (peak_obs == (len(obs) - i_end)) and (i_end < len(obs)):
                peak_obs = _np.argmax(obs[:-i_end])
                i_end += 1
            
            
            true_obs = dxdt[start_ + peak_obs: stop_]

            true_obs = create_signal(abs(true_obs),
                                     sampling_freq=fsamp,
                                     start_time=times[start_ + peak_obs])

            # find the 'first minimum' (zero) the derivative (peak)
            minima = _Minima(win_len=0.1, win_step=0.025,
                             method='windowing')(true_obs)

            idx_mins = _np.where(~_np.isnan(minima.p.main_signal.values))[0].ravel()

            if len(idx_mins) >= 1:
                peak = idx_mins[0]
                true_peaks.append(start_ + peak_obs + peak + 1)
            # else:
            #     print('Peak not found; idx_beat: ' + str(idx_beat))
            #     pass
        true_peaks = _np.array(true_peaks)
        
        # STAGE 3 - FINALIZE computing IBI
        ibi_scaffold = _np.nan * _np.zeros(len(signal.values))
        
        if len(true_peaks)>1:
            t_ibi = true_peaks / fsamp
            v_ibi = _np.diff(t_ibi)
        
            v_ibi = _np.insert(v_ibi, 0, v_ibi[0])

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
        self.dimensions = {'time': 0}

    def algorithm(self, signal):
        params = self._params
        bpm_max, delta, k = params["bpm_max"], params["delta"], params["k"]
        fmax = bpm_max / 60

        fsamp = signal.p.get_sampling_freq()

        if delta == 0:
            delta = k * _SignalRange(win_len=2 / fmax,
                                     win_step=0.5 / fmax, smooth=False)(signal)
            delta = _np.array(delta).ravel()

        # adjust for delta values equal to 0
        idx_delta_zeros = _np.where(delta == 0)[0]
        idx_delta_nozeros = _np.where(delta > 0)[0]
        delta[idx_delta_zeros] = _np.min(delta[idx_delta_nozeros])

        refractory = 1 / fmax

        # find beats
        maxp = _PeakDetection(
            delta=delta, refractory=refractory)(signal)
        maxp = _np.array(maxp).ravel()

        if maxp[0] == 0:
            maxp = maxp[1:]

        idx_beats = _np.where(~_np.isnan(maxp))[0]

        times_beats = idx_beats / fsamp

        ibi_values = _np.diff(times_beats)

        ibi_values = _np.insert(ibi_values, 0, ibi_values[0])

        ibi_scaffold = _np.nan * _np.zeros(len(signal.values))

        ibi_scaffold[idx_beats] = ibi_values

        return ibi_scaffold

class RemoveBeatOutliers(_Algorithm):
    """
    Detect and remove outliers in the IBI signal. 

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
    ibi : pyphysio.Signal
        Corrected ibi

    """

    def __init__(self, ibi_median=0, cache=3, sensitivity=0.25):
        assert ibi_median >= 0, "IBI median value should be positive (or equal to 0 for automatic computation"
        assert cache >= 1, "Cache size should be greater than 1"
        assert sensitivity > 0, "Sensitivity value shlud be positive"

        _Algorithm.__init__(self, ibi_median=ibi_median,
                            cache=cache, sensitivity=sensitivity)
        self.dimensions = {'time': 0}

    def algorithm(self, signal):
        assert signal.p.get_sampling_freq() != 'unevenly', "This algorithm should be applied to evenly IBI. Avoid processing nans before"
        
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

    def __init__(self, ibi_median=0, cache=3, sensitivity=0.25):
        assert ibi_median >= 0, "IBI median value should be positive (or equal to 0 for automatic computation"
        assert cache >= 1, "Cache size should be greater than 1"
        assert sensitivity > 0, "Sensitivity value shlud be positive"

        _Algorithm.__init__(self, ibi_median=ibi_median,
                            cache=cache, sensitivity=sensitivity)
        self.dimensions = {'time': 0}


    def _add_peaks(self, t_prev, t_curr, ibi_cache, bvp_signal=None):
        params = self._params
        sensitivity = params["sensitivity"]
            
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
    
    def algorithm(self, signal, bvp_signal = None):
        assert signal.p.get_sampling_freq() != 'unevenly', "This algorithm should be applied to evenly IBI. Avoid processing nans before"
        
        params = self._params
        cache, sensitivity, ibi_median = params["cache"], \
            params["sensitivity"], params["ibi_median"]
            
        fsamp = signal.p.get_sampling_freq()
        
        ibi_values = signal.p.get_values().ravel()
        idx_values = _np.where(~_np.isnan(ibi_values))

        ibi_values = ibi_values[idx_values]
        t_ibi = signal.p.get_times()[idx_values]

        if ibi_median == 0:
            ibi_expected = _np.median(ibi_values)
        else:
            ibi_expected = ibi_median

        #% RUN FORWARD CORRECTION
        ibi_cache = _np.repeat(ibi_expected, cache)
        counter_bad = 0

        prev_t_ibi = t_ibi[0]
        t_ibi_1 = [prev_t_ibi]

        for id_ibi in _np.arange(1, len(t_ibi)):
            curr_t_ibi = t_ibi[id_ibi]
            curr_median = _np.median(ibi_cache)
            curr_ibi = curr_t_ibi - prev_t_ibi
            
            if curr_ibi > curr_median * (1 + sensitivity):  
                # abnormal peak: probably a missing beat
                counter_bad += 1
                
                #we assume there are missing beat(s) in between
                t_missed_peaks = self._add_peaks(prev_t_ibi, curr_t_ibi, ibi_cache, bvp_signal)
                for t in t_missed_peaks:
                    t_ibi_1.append(t) 
                
                t_ibi_1.append(curr_t_ibi)
                prev_t_ibi = curr_t_ibi
                
            elif curr_ibi < curr_median * (1 - sensitivity):
                # abnormal peak: probably a false beat
                counter_bad += 1
                
            else:
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                t_ibi_1.append(curr_t_ibi)
                prev_t_ibi = curr_t_ibi
                
            if counter_bad == cache:  
                # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                counter_bad = 0

        # RUN BACKWARD CORRECTION
        prev_t_ibi = t_ibi[-1]
        t_ibi_2 = [prev_t_ibi]

        for id_ibi in _np.arange(len(t_ibi)-2,-1,-1): #idx go backward
            curr_t_ibi = t_ibi[id_ibi]
            curr_median = _np.median(ibi_cache)
            curr_ibi = abs(curr_t_ibi - prev_t_ibi)
            
            if curr_ibi > curr_median * (1 + sensitivity): 
                # abnormal peak: probably a missing beat
                counter_bad += 1
                
                #we assume there are missing beat(s) in between
                #note: we change the order of curr_ibi and prev_ibi 
                #as we are going backward                
                t_missed_peaks = self._add_peaks(curr_t_ibi, prev_t_ibi, ibi_cache, bvp_signal)
                for t in t_missed_peaks:
                    t_ibi_2.append(t) 
                    
                t_ibi_2.append(curr_t_ibi)
                prev_t_ibi = curr_t_ibi
                
            elif curr_ibi < curr_median * (1 - sensitivity):
                # abnormal peak: probably a false beat
                counter_bad += 1
            
            else:
                ibi_cache = _np.r_[ibi_cache[1:], curr_ibi]
                t_ibi_2.append(curr_t_ibi)
                prev_t_ibi = curr_t_ibi
                
            if counter_bad == cache:  
                # ibi cache probably corrupted, reinitialize
                ibi_cache = _np.repeat(ibi_expected, cache)
                counter_bad = 0

        t_ibi_1 = _np.array(t_ibi_1)
        t_ibi_2 = _np.array(t_ibi_2)[::-1]

        # PAIR BEATS
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

        # COMBINATORIAL EXPLORATION
        #TODO: see issues with long portions; try to think of a 
        # greedy algorithm that select the best ibi starting from the start 
        #to the end? Maybe we can exploit forw. and backw. direction again
        
        # define zones where there are different values
        diff_idxs = pairs[:, 0] - pairs[:, 1]
        diff_idxs[diff_idxs != 0] = 1
        diff_idxs = _np.diff(diff_idxs)

        starts = _np.where(diff_idxs == 1)[0]+1
        stops = _np.where(diff_idxs == -1)[0]

        if len(starts)==0: 
            # no differences
            t_out = t_ibi_1

        else:
            #adjust starts and stops
            if len(stops)==0:
                stops = _np.array([starts[-1] + 1])
                
            if starts[0] >= stops[0]:
                stops = stops[1:]
            
            stops += 1
            
            if len(starts) > len(stops):
                stops = _np.r_[stops, starts[-1] + 1]
            
            #each start should have a corresponding stop distant at least 1
            assert sum((stops-starts)<1) == 0
            
            #compose output t ibi
            t_out = _np.copy(pairs[:, 0])
            
            #for all portions in which 
            #beats from forw. and bacw. run are different
            for i in _np.arange(len(starts)):
                
                #find start and end portion
                i_st = starts[i]
                i_sp = stops[i]
               
                if i_sp > len(t_out) - 1:
                    i_sp = len(t_out) - 1
                
                
                # NOTE
                # combinatorial exploration of long portions is computationally
                # demanding (exponential!)
                # we therefore need to partition long portions
                
                # if the length of the portion is <= 10 beats, that's fine
                # do not partition
                if (i_sp - i_st) <= 10:
                    i_st_ = [i_st]
                    i_sp_ = [i_sp]
                
                else:
                    #partition the long portion
                    n_ = int(_np.round((i_sp - i_st) / 10)) 
                    idx_cuts = _np.linspace(i_st, i_sp, n_+1).astype(int)
                    
                    i_st_ = idx_cuts[:-1]
                    i_sp_ = idx_cuts[1:]
                    
                #run combinatorial exploration on each partition
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
                        #TODO: is SD a good error measure to be minimized?
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
        
        return(ibi_scaffold)
