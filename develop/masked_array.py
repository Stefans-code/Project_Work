# coding=utf-8
import numpy as _np
import numpy.ma as _ma

class Signal(_ma.MaskedArray):    
    def __new__(self, values, sampling_freq=1, start_time = 0, info={}, mask = None):
        if mask is None:
            mask = _np.zeros_like(values).astype(bool)
        
        obj = _ma.array(data=values, mask=mask).view(self)
        obj._pyphysio = {'sampling_frequency': sampling_freq,
                         'start_time': start_time,
                         'info': info}
        obj._mask = mask
        obj._fill_value = _np.nan
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        
        if hasattr(obj, '_pyphysio'):
            self._pyphysio = getattr(obj, '_pyphysio')
        else:
            self._pyphysio = {'sampling_frequency': 1,
                              'start_time': 0,
                              'info': {}}
            
        if hasattr(obj, '_mask'):
            self._mask = getattr(obj, '_mask')
        else:
            self._mask = False
            
        if hasattr(obj, '_fill_value'):
            self._fill_value = getattr(obj, '_fill_value')
        else:
            self._fill_value = _np.nan
    
    @property
    def ph(self):
        return self._pyphysio
    
    def get_sampling_freq(self):
        return self.ph['sampling_frequency']

    def set_sampling_freq(self, value):
        self.ph['sampling_frequency'] = value
    
    def get_start_time(self):
        return self.ph['start_time']

    def set_start_time(self, value):
        self.ph['start_time'] = value    
    
    def get_info(self):
        return self.ph['info']

    def set_info(self, value):
        self.ph['info'] = value
        
    def __getitem__(self, item):
        sampling_frequency = self._pyphysio['sampling_frequency']
        start_time = self._pyphysio['start_time']
        info = self._pyphysio['info']

        #######################
        # process values
        #######################
        values = self.data
        mask = self.mask
        selected_values = values.__getitem__(item)
        selected_mask = mask.__getitem__(item)
        
        #If we are selecting an index on all axis 
        #then we are extracting only one scalar:
        #just return the scalar and avoid processing the attributes
        #(This is also to avoid issues with IDE variable viewers)
        if isinstance(item, tuple):
            print(1)
            if _np.array([isinstance(x, int) for x in item]).all():
                return _ma.MaskedArray(data = selected_values,
                                       mask = selected_mask,
                                       fill_value = self._fill_value)
        
        #If we are selecting on 0 axis using a list/ndarray
        #then we lose the temporal dimension
        #just return the selected masked array
        if isinstance(item, tuple):
            print('1b')
            if (isinstance(item[0], list)) or (isinstance(item[0], _np.ndarray)):
                return _ma.MaskedArray(data = selected_values,
                                       mask = selected_mask,
                                       fill_value = self._fill_value)
            
        #1- fixing dimensions
        #If selecting only one timepoint (=int on 0 axis)
        #maintain original number of dimensions
        if isinstance(item, tuple):
            print(2)
            if isinstance(item[0], int):
                print(3)
                selected_values = _ma.expand_dims(selected_values, 0)
                selected_mask = _ma.expand_dims(selected_mask, 0)

        #If selecting only one timepoint (=int on 0 axis)
        #(but no slicing on other axes)
        #maintain original number of dimensions
        if isinstance(item, int):
            print(4)
            selected_values = _ma.expand_dims(selected_values, 0)
            selected_mask = _ma.expand_dims(selected_mask, 0)
        
        
        #######################
        # process temporal metadata
        #######################
        
        #separate item for the first axis from others
        if isinstance(item, tuple): #more than one axis involved
            item_0 = item[0]
        else:
            item_0 = item
        
        print('--')
        print(item_0)
        print(self.shape)
        print(type(item_0))
        #set start time and new sampling freq
        if isinstance(item_0, int):
            new_start_time = start_time + item_0/sampling_frequency
            new_sampling_freq = sampling_frequency
        elif item_0 is Ellipsis:
            new_start_time = start_time
            new_sampling_freq = sampling_frequency
        elif item_0 is None:
            new_start_time = start_time
            new_sampling_freq = sampling_frequency
        else: #slice
            start = item_0.start if item_0.start is not None else 0
            new_start_time = start_time + start/sampling_frequency
            
            ratio = item_0.step if item_0.step is not None else 1
            new_sampling_freq = sampling_frequency/ratio
        
        ######################
        # finalize
        selected = self.__class__(selected_values, 
                                  new_sampling_freq,
                                  new_start_time,
                                  info,                                  
                                  mask = selected_mask)
        
        #process info
        #selected = self.__getitem_attrib__(selected, item)
        
        return selected
    
    def __getitem_attrib__(self, selected, item):
        
        
        #separate item for the first axis from others
        item_other = None
        
        if isinstance(item, tuple): #more than one axis involved
            item_0 = item[0]
            item_other = item[1:]
        else:
            item_0 = item
        
        #=========================
        # work on metadata in info dict
        info = selected.get_info()
        #set sqi if existing
        #set good if existing
        #sqi and good should be changed only if working on other dims
        if isinstance(item_other, tuple):
            
            #It ONLY manages the channels and components
            #TODO: manage the first axis (time)
            
            #create a new item that ignores the first dimension
            item_new = (slice(None,None,None), *item_other)
            
            if 'sqi' in info.keys():
                #TODO: MANAGE SQI LIST
                #idea: use 4th axis instead of list?
                
                #otherwise it will probably throw an error
                #apply the item_new to the sqi
                new_sqi = {}
                for s in info['sqi'].keys():
                    sqi = info['sqi'][s].clone()
                    new_sqi[s] = sqi.__getitem__(item_new)
                    
                selected.update_info('sqi', new_sqi)
                
            if 'good' in info.keys():
                # new_good = {}
                # for s in info['good'].keys():
                new_good = info['good'].__getitem__(item_new)
                selected.update_info('good', new_good)
        
        return selected
    
    def __repr__(self):
        return super().__repr__()


class UnevenlySignal(Signal):
    def __new__(self, values, sampling_freq=1, start_time = 0, info={}, mask=None,
                x_values = None, x_type='indices'):
        
        #for compatibility
        if mask is not None:
            assert x_values is None, "When defining UE with mask, x_values should be none"
            obj = _ma.array(data=values, mask=mask).view(self)
            obj._pyphysio = {'sampling_frequency': sampling_freq,
                             'start_time': start_time,
                             'info': info}
            obj._mask = mask
            obj._fill_value = _np.nan
            return obj
        
        assert x_values is not None, "x_values are missing"
        assert x_type in ['indices', 'instants'], "x_type not in ['indices', 'instants']"
        
        assert len(x_values) == len(values), "Length mismatch (y:%d vs. x:%d)" % (len(values), len(x_values))
        assert len(_np.where(_np.diff(x_values) <= 0)[0]) == 0, 'Given x_values are not strictly monotonic'
        
        if x_type == 'instants':
            assert start_time<= x_values[0], 'the first instant is before the start_time'
        
        #compute the size of the supporting EvenlySignal
        if x_type == 'indices':
            size_0 = x_values[-1] + 1
        else:
            size_0 = int(_np.ceil(sampling_freq*(x_values[-1] - start_time)))+1
        
        #compute non masked indices
        if x_type == 'indices':
            idx_0 = x_values
        else:
            idx_0 = _np.round(sampling_freq*(x_values - start_time)).astype(int)
            
        values_shape = list(values.shape)
        new_shape = values_shape
        new_shape[0] = size_0
        new_shape = tuple(new_shape)
        
        #create values of supporting EvenlySignal
        new_values = _np.empty(new_shape)
        new_values[idx_0] = values
        
        #create mask
        mask = _np.ones(new_shape)
        mask[idx_0] = 0
        mask = mask.astype(bool)
        
        print(new_values.shape)
        print(mask.shape)
        
        obj = _ma.array(data=new_values, mask=mask).view(self)
        obj._pyphysio = {'sampling_frequency': sampling_freq,
                         'start_time': start_time,
                         'info': info}
        obj._mask = mask
        obj._fill_value = _np.nan
        return obj

    def get_indices(self):
        selection = list(_np.zeros(self.ndim).astype(int))
        selection[0] = slice(None, None, None)
        selection = tuple(selection)
        # print(selection)
        mask_0 = self._mask.__getitem__(selection)
        indices = _np.where(~mask_0)[0]
        return indices

class EvenlySignal(Signal):
    def __new__(self, values, sampling_freq=1, start_time=0, info={}, mask=None):
        assert mask is None, "mask should be none when creating an EvenlySignal"
        obj = _ma.array(data=values, mask=None).view(self)
        obj._pyphysio = {'sampling_frequency': sampling_freq,
                         'start_time': start_time,
                         'info': info}
        obj._mask = mask
        obj._fill_value = _np.nan
        
#%%
values = _np.random.uniform(size=(1000,3,5))
info = {'ciao': 'ciao'}
x_values = _np.arange(0, 2000, 2)
x_type='instants'

signal = EvenlySignal(values, 1, 0, info)

# #%%

# mask = _np.zeros_like(values).astype(bool)
# mask[200:300,:,:] = True

# masked = _ma.array(values, mask = mask)

# #%%
# signal = Signal(values, 1, 0, info, mask=mask)
indices = signal.get_indices()
#%%
print('signal')
s = signal[100:]
print(s._pyphysio)
print(s.data.shape)
print(s.mask.shape)
print(s.mask.sum())

#%%
print('signal')
s = signal[250:]
print(s._pyphysio)
print(s.data.shape)
print(s.mask.shape)
print(s.mask.sum())

#%%
print('signal')
s = signal[[0,10,234,250],:]
print(s.data.shape)
print(s.mask.shape)
print(s.mask.sum())

#%%
print('signal')
s = signal[200, 0, 0]
print(s)

#%%
print('signal')
print(_np.median(signal))
print(_np.median(signal.data))

#%%
print(_ma.median(signal))
print(_ma.median(signal.data))

#%%
print('signal')
s = _ma.apply_along_axis(_np.mean, 0, signal)
print(s.data.shape)
print(s.mask.shape)
print(type(s))

#%%
def normalize(x):
    return(x - _np.mean(x))

print('signal')
s = _ma.apply_along_axis(normalize, 0, signal)
print(s.data.shape)
print(s.mask.shape)
print(type(s))
# print(s.mask)

#%%