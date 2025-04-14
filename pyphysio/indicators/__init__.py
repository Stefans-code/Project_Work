#TODO: is it correct to do so?
# from  . import frequencydomain as fd
# from . import timedomain as td
# from . import nonlinear as nl
# from . import peaks as pk
from .._base_algorithm import _Algorithm

class _Indicator(_Algorithm):
    def __init__(self, **kwargs):
        _Algorithm.__init__(self, **kwargs)
        self.required_dims = ['time']
        
    def __get_template__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        t_out = signal['time'][0]
        template = self.__compute_template__(signal, {'time': [t_out]})
        return(chunk_dict, template)

def compute_indicators(indicators, signal):
    indicators_dict = {}
    
    for ind in indicators:
        indicators_dict[ind.get('name')] = ind(signal).p.get_values()[0][0][0]
        
    return(indicators_dict)