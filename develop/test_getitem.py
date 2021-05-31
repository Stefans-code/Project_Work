import numpy as _np
from numbers import Number as _Number
import copy

from test_utils import info

class Signal(_np.ndarray):
    
    def __new__(cls, values, sampling_freq, start_time=None, info = {}):
        assert sampling_freq > 0, "The sampling frequency cannot be zero or negative"
        assert start_time is None or isinstance(start_time, _Number), "Start time is not numeric"
        
        #convert values to ndarray
        values = _np.asarray(values)
        
        obj = _np.asarray(values).view(cls)
        
        obj._pyphysio = {
            'sampling_freq': sampling_freq,
            'start_time': start_time if start_time is not None else 0,
            'info': info
        }
        
        # setattr(obj, "_mutated", False)
        return obj

    def __array_finalize__(self, obj):
        # __new__ called if obj is None
        if obj is not None and hasattr(obj, '_pyphysio'):
            self._pyphysio = getattr(obj, '_pyphysio').copy()

    def __array_wrap__(self, out_arr, context=None):
        # Just call the parent's
        # noinspection PyArgumentList
        if isinstance(out_arr, Signal):
            return _np.ndarray.__array_wrap__(self, out_arr, context)
        else:
            return out_arr
        
    def get_values(self):
        return _np.asarray(self)
    
    def clone(self):
        obj = self.copy()
        obj._pyphysio = copy.deepcopy(self.ph)
        return(obj)
    
    def clone_properties(self, new_values):
        x_new = self.__class__(new_values,
                               self.get_sampling_freq(),
                               self.get_start_time(),
                               self.get_info())
        return(x_new)
    
    @property
    def ph(self):
        return self._pyphysio
    
    def get_sampling_freq(self):
        return self.ph['sampling_freq']

    def set_sampling_freq(self, value):
        self.ph['sampling_freq'] = value
    
    def get_start_time(self):
        return self.ph['start_time']

    def set_start_time(self, value):
        self.ph['start_time'] = value    
    
    def get_info(self):
        return self.ph['info']

    def set_info(self, value):
        self.ph['info'] = value    
        
    def __getitem__(self, item):
        # print(item, self.shape)
        #TODO if float segment based on time

        #apply __getitem__ to values (ndarray)
        values = self.get_values()
        selected_values = values.__getitem__(item)
        
        #try to catch Ellipsis issue with the np.apply_along_axis
        #which uses ellipsis and changes the shape
        if isinstance(item, tuple) and Ellipsis in item:
            return(self.clone_properties(selected_values))
        
        #If we are selecting an index on all axis 
        #then we are extracting only one scalar:
        #just return the scalar and avoid processing the attributes
        #(This is also to avoid issues with IDE variable viewers)
        if isinstance(item, tuple):
            if _np.array([isinstance(x, int) for x in item]).all():
                return selected_values
                
        selected = self.clone_properties(selected_values)
        
        #but this is a signal, so we need additional steps
        #to ensure the result is still a valid signal
        
        #1- processing attributes
        #process metadata
        selected = self.__getitem_attrib__(item, selected)
        
        #2- fixing dimensions
        #If selecting only one timepoint (=int on 0 axis)
        #maintain original number of dimensions
        if isinstance(item, tuple):
            if isinstance(item[0], int):
                selected = _np.expand_dims(selected, 0)
        
        #If selecting only one timepoint (=int on 0 axis)
        #(but no slicing on other axes)
        #maintain original number of dimensions
        if isinstance(item, int):
            selected = _np.expand_dims(selected, 0)
        
        return selected
    
    def __getitem_attrib__(self, item, selected):
        selected = selected.clone()
        original_sampling_freq = selected.get_sampling_freq()
        original_start_time = selected.get_start_time()
        
        #separate item for the first axis from others
        item_other = None
        
        if isinstance(item, tuple): #more than one axis involved
            item_0 = item[0]
            item_other = item[1:]
        else:
            item_0 = item
        
        #set start time and new sampling freq
        if isinstance(item_0, int):
            new_start_time = original_start_time + item_0/original_sampling_freq
            new_sampling_freq = original_sampling_freq
        elif item_0 is Ellipsis:
            new_start_time = original_start_time
            new_sampling_freq = original_sampling_freq
        elif item_0 is None:
            new_start_time = original_start_time
            new_sampling_freq = original_sampling_freq
        else: #slice
            start = item_0.start if item_0.start is not None else 0
            new_start_time = original_start_time + start/original_sampling_freq
            
            ratio = item_0.step if item_0.step is not None else 1
            new_sampling_freq = original_sampling_freq/ratio
        
        selected.set_start_time(new_start_time)
        selected.set_sampling_freq(new_sampling_freq)

        return selected    
    
#%%
fsamp = 10
start_time = 0
values = _np.random.uniform(size=(1000,5,3))

signal = Signal(values, fsamp, start_time)
info(signal)

#%%
print(_np.median(signal))

#%%
mean = _np.apply_along_axis(_np.mean, 0, signal)
print(mean.shape)

#%%
def norm(x):
    return(x - _np.mean(x))

norm_sig = _np.apply_along_axis(norm, 0, signal)
print(norm_sig.shape)

#%%
signal_ = signal[:100]
info(signal_)

#%%
signal_ = signal[:100,0]
info(signal_)

#%%
signal_ = signal[:100,0,0]
info(signal_)

#%%
signal_ = signal[100:480,0,0]
info(signal_)

#%%
signal_ = signal[100,0,0]
print(type(signal_))

#%%
signal_ = signal[100]
info(signal_)

#%%
signal_ = signal[::2, 3]
info(signal_)
