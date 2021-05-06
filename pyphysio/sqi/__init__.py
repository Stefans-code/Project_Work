import numpy as _np
from ..processing import Algorithm as _Algorithm
from ..signal import Signal as _Signal
from ..segmenters import _Segmenter,\
    FixedSegments as _FixedSegments

from ..segmenters import fmap as _fmap

class SignalQualityIndicator(_Algorithm):
    """ 
    A Signal Quality Indicator is a special class of indicators
    that also returns if the value is within a range.
    Used to check the quality of signals.
    """
    def __init__(self, threshold, **kwargs):
        assert len(threshold)==2
        _Algorithm.__init__(self, threshold=threshold, **kwargs)
        
    def is_good(self, sqi_values):
        params = self._params
        threshold = params['threshold']
        if sqi_values.ndim == 0:
            output = (sqi_values >= threshold[0]) & (sqi_values <= threshold[1])
            output = _np.array(output)
        else:
            output = _np.zeros_like(sqi_values)
            idx_good = _np.where((sqi_values >= threshold[0]) & (sqi_values <= threshold[1]))
            output[idx_good] = 1
        return(output)
        
    def __call__(self, data):
        """
        Executes the algorithm using the parameters saved by the constructor.
        @param data: The data.
        @type data: TimeSeries
        @return: The result.
        """
        values_out = super().__call__(data)
        
        isgood = self.is_good(values_out)
        return(values_out, isgood)

class ComputeQuality(_Algorithm):
    def __init__(self, sqi, segmenter=None, **kwargs):
        assert len(sqi) > 0
        for sqi_ in sqi:
            assert isinstance(sqi_, SignalQualityIndicator)

        assert segmenter is None or isinstance(segmenter, _Segmenter)
        _Algorithm.__init__(self, sqi=sqi, segmenter=segmenter, **kwargs)
        
    
    def algorithm(self, signal):
        params = self._params
        
        #if no segmentation required, create a segmenter with a unique segment
        segmenter = params['segmenter']
        sqi = params['sqi']
        
        if segmenter is None: 
            segmenter = _FixedSegments(signal.get_duration(), 
                                       drop_cut=False, 
                                       drop_mixed=False)
        
        #COMPUTE SQI
        sqi_values = _fmap(segmenter, sqi, signal)
        
        #COMPUTE GOOD SIGNALS
        
        #this should return a signal with updated sqi and good_signal in info
        return(sqi_values)
    
    # def compute_good_sqi(nirs):
    #     assert 'sqi' in nirs.info.keys(), "SQI not computed, please run 'compute_sqi' first"
    #     sqi_values = nirs.info['sqi']
    #     sqi_indicators = nirs.info['sqi_indicators']
        
    #     #GET GOOD VALUES
    #     #initialize output matrix
    #     sqi_good = np.zeros_like(sqi_values).astype(bool)
    #     sqi_good[:,0:3, :] = True #set segment data to True
        
    #     #for all sqi
    #     for i_sqi, sqi in enumerate(sqi_indicators.keys()):
    #         th = sqi_thresholds[sqi]
    #         curr_sqi_values = sqi_values[:, i_sqi+3, :]
    #         idx_good = np.where((curr_sqi_values>= th[0]) & (curr_sqi_values <= th[1]))
    #         sqi_good[:, i_sqi+3,:][idx_good] = True
        
    #     return(sqi_good)
    
    # def compute_good_channels(nirs, ratio=0.9):
    #     assert 'sqi' in nirs.info.keys(), "SQI not computed, please run 'compute_sqi' first"
        
    #     sqi_good = compute_good_sqi(nirs)
        
    #     idx_good_channels = []
    #     for i_ch in range(sqi_good.shape[2]): #for all channels
    #         sqi_channel = sqi_good[:, :, i_ch]    
    #         n_good = np.sum(sqi_channel, axis = 0)
            
    #         ratio_good = n_good / sqi_channel.shape[0]
    
    #         if (ratio_good >= ratio).all():
    #             idx_good_channels.append(i_ch)
        
    #     good_channels = np.repeat(False, sqi_good.shape[2])
    #     good_channels[idx_good_channels] = True
    #     nirs.info['good_channels'] = good_channels
    #     return(nirs)