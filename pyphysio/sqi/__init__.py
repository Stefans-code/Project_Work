import numpy as _np
import xarray as _xr
from ..processing import Algorithm as _Algorithm

_xr.set_options(keep_attrs = True)

class SignalQualityIndicator(_Algorithm):
    """ 
    A Signal Quality Indicator is a special class of indicators
    that also returns if the value is within a range.
    Used to check the quality of signals.

    Args:
        threshold (low, high): The range within which the sqi indicates good quality
    
    Returns:
        result (sqi, isgood): Tuple containing the value of the sqi and if it corresponds to good quality
    
    """
    def __init__(self, threshold, **kwargs):
        '''
        '''
        assert len(threshold)==2
        _Algorithm.__init__(self, threshold=threshold, **kwargs)
    
    def is_good(self, sqi_values):
        # print('-----> is_good')
        params = self._params
        threshold = params['threshold']
        if sqi_values.ndim == 0:
            output = (sqi_values >= threshold[0]) & (sqi_values <= threshold[1])
            output = _np.array(output)
        else:
            output = _np.zeros_like(sqi_values)
            idx_good = _np.where((sqi_values >= threshold[0]) & (sqi_values <= threshold[1]))
            output[idx_good] = 1
        
        # print('<----- is_good')
        return(output)
        
    def __call__(self, signal, add_signal=True, dimensions=None):
        # print('-----> SQI.__call__()')
        values_out = super().__call__(signal, add_signal=add_signal, 
                                      dimensions=dimensions)
        
        signal_name = signal.p.main_signal.name
        if add_signal:
            indicator_name = signal_name+'_'+self.name
        else:
            indicator_name = signal_name
        
        #for SQI that are called from within other algorithms
        if isinstance(values_out, _xr.Dataset):
            isgood = self.is_good(values_out[indicator_name])
            #convert isgood to dataarray
            isgood_out = values_out[indicator_name].copy(data = isgood)
        else:
            values_out.name = indicator_name
            isgood = self.is_good(values_out)
            #convert isgood to dataarray
            isgood_out = values_out.copy(data = isgood)
        
        
        isgood_name = indicator_name +'_isgood'
        isgood_out.name = isgood_name
        
        out = _xr.merge([values_out, isgood_out])
        # print('<----- SQI.__call__()')
        return(out)

def compute_good_global(is_good, ratio):
    #at leat ratio% timepoints should be good
    is_good_ = is_good.copy()
    is_good_ = _np.sum(is_good_, axis=0, keepdims=True) >= ratio*is_good_.shape[0]
    return(is_good_)

"""
class ComputeQuality(_Algorithm):
    '''
    Automitize the computation of SQI and the decision about the overall signal quality of a signal.
    When called on a signal it returns the same signal with added information about the signal quality.
    
    
    @param sqi: The Signal Quality Indicators to be computed
    @type sqi: List of Signal Quality Indicators
    
    @param segmenter: (optional) a Segmenter. If provided, the SQI will be computed on each segment
    
    @param compute_global: When a segmenter is provided,
                           defines how to assess if the signal is good.
                           If True (default) the quality of the signal is assessed
                           if False the quality is assessed by segment.
    @type compute_global: bool
    
    @param ratio: float between 0 and 1, (if segmenter is provided and compute_global = True)
                  portion of segments that should have good SQI to decide that the signal is good 
                  
    @return: Signal with added information: 'sqi' the computed SQI, 'good' the decisin about the signal quality.
    '''
    def __init__(self, sqi, segmenter=None, compute_global=True, ratio=1, **kwargs):
        
        assert len(sqi) > 0
        for sqi_ in sqi:
            assert isinstance(sqi_, SignalQualityIndicator)
        
        assert ratio>=0 and ratio <=1
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
        
        mask = is_good_[0].mask
        
        #COMPUTE GOOD
        #---------
        #first, to be good, all sqi should be good
        
        #stack over a new 0 axis
        is_good_np = _np.stack([x.get_values() for x in is_good_], axis=0)
        #is good if all sqi (on the 0 axis) are good
        #i.e. the sum is equal to the number of sqi
        is_good_np = _np.sum(is_good_np, axis=0) == is_good_np.shape[0]
        
        sqi_key = list(sqi_values.keys())[0]
        #---------
        #now, decide whether to get global or local indications
        if compute_global:
            is_good_global = compute_good_global(is_good_np, ratio)
            is_good_signal = sqi_values_[sqi_key].clone_properties(is_good_global)
            signal.update_info('good', is_good_signal)
                
        else:
            
            is_good_signal = sqi_values_[sqi_key].clone_properties(is_good_np, 
                                                                   x_values = _np.where(mask[:,0,0] == False)[0],
                                                                   x_type = 'indices')
            signal.update_info('good', is_good_signal)
        
        
        #return a signal with updated 'sqi' and 'good' in info
        return(signal)
    
    def __repr__(self):
        return super(ComputeQuality, self).__repr__()
"""    