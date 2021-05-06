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
    def __init__(self, sqi, segmenter=None, compute_global=True, ratio=1, **kwargs):
        assert len(sqi) > 0
        for sqi_ in sqi:
            assert isinstance(sqi_, SignalQualityIndicator)

        assert segmenter is None or isinstance(segmenter, _Segmenter)
        _Algorithm.__init__(self, sqi=sqi, segmenter=segmenter,
                            compute_global=compute_global, ratio=ratio, **kwargs)
        
    
    def __call__(self, signal):
        params = self._params
        
        segmenter = params['segmenter']
        sqi = params['sqi']
        compute_global = params['compute_global']
        ratio = params['ratio']
        
        #if no segmentation required, create a segmenter with a unique segment
        if segmenter is None: 
            segmenter = _FixedSegments(signal.get_duration(), 
                                       drop_cut=False, 
                                       drop_mixed=False)
        
        #COMPUTE SQI
        sqi_values = _fmap(segmenter, sqi, signal)
        
        #get sqi_values and is_good for each sqi
        sqi_values_ = {}
        is_good_ = []
        for k,v in sqi_values.items():
            sqi_values_[k] = v[0]
            is_good_.append(v[1])
        
        #save sqi only in signal.info
        signal.update_info('sqi', sqi_values_)
        
        #COMPUTE GOOD
        #---------
        #first, to be good, all sqi should be good
        
        #stack over a new 0 axis
        is_good_ = _np.stack(is_good_, axis=0)
        #is good if all sqi (on the 0 axis) are good
        #i.e. the sum is equal to the number of sqi
        is_good_ = _np.sum(is_good_, axis=0) == is_good_.shape[0]
        
        #---------
        #now, decide whether to get global or local indications
        
        #if only one timepoint, then it is global
        if is_good_.shape[0] == 1:
            signal.update_info('good', is_good_)
        
        else:
            if compute_global:
                #at leat ratio% timepoints should be good
                is_good_ = _np.sum(is_good_, axis=0, keepdims=True) >= ratio*is_good_.shape[0]
                signal.update_info('good', is_good_)
                
            else:
                signal.update_info('good', is_good_)
        
        
        #return a signal with updated 'sqi' and 'good' in info
        return(signal)
    
    def __repr__(self):
        return super(ComputeQuality, self).__repr__()