#TODO: is it correct to do so?
from  . import frequencydomain as fd
from . import timedomain as td
from . import nonlinear as nl
from . import peaks as pk



def compute_indicators(indicators, signal):
    indicators_dict = {}
    
    for ind in indicators:
        indicators_dict[ind.get('name')] = ind(signal).p.get_values()[0][0][0]
        
    return(indicators_dict)