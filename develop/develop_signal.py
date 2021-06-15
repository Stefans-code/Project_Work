# coding=utf-8
import numpy as _np
import numpy.ma as _ma
import copy

class Signal(_ma.MaskedArray):
    def __new__(self, 
                data, 
                sampling_freq=1, 
                start_time = 0,
                info={}, 
                mask=None):
        if mask is None:
            mask = _np.zeros_like(data).astype(bool)

        obj = _ma.MaskedArray(data=data, mask=mask,
                              subok=False, hard_mask=False)
        
        obj._optinfo = {'sampling_freq': sampling_freq,
                        'start_time': start_time,
                        'info': info}
        
        obj = obj.view(self)
        return obj

    # def __array_finalize__(self, obj):
    #     if obj is None: return
                
    #     mask = _np.zeros_like(len(self)).astype(bool)
    #     self._mask = getattr(obj, '_mask', mask)
    #     self._hardmask = getattr(obj, '_hardmask', False)
    #     self._fill_value = getattr(obj, '_fill_value', _np.nan)
        
    #     if hasattr(obj, '_optinfo'):
    #         self._optinfo = getattr(obj, '_optinfo')
    #     else:
    #         print('no _optinfo')

    
    # def __array_wrap__(self, out_arr, context=None):
    #     print('In __array_wrap__:')
    #     print('   self is %s' % repr(self))
    #     print('   arr is %s' % repr(out_arr))
    #     # then just call the parent
    #     return super(Signal, self).__array_wrap__(self, out_arr, context)
    
    # def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
    #     print('In __array_ufunc__:')
    #     print('   self is %s' % repr(self))
    #     print('   arr is %s' % repr(out))
    #     # then just call the parent
    #     return super(Signal, self).__array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs)
    
    def clone_properties(self, new_data, new_mask=None):
        x_new = Signal(new_data,
                       self.ph['sampling_freq'],
                       self.ph['info'],
                       new_mask)
        return(x_new)
        
    def __getitem__(self, item):
        print(item)
        sampling_frequency = self.ph['sampling_freq']
        start_time = self.ph['start_time']
        info = self.ph['info']

        #######################
        # process values
        #######################
        values = self.data
        mask = self.mask
        selected_values = values.__getitem__(item)
        selected_mask = mask.__getitem__(item)
        
        if isinstance(item, tuple): #more than one axis involved
            item_0 = item[0]
        else:
            item_0 = item
            
        #If we are selecting an index on all axis 
        #then we are extracting only one scalar:
        #just return the scalar and avoid processing the attributes
        #(This is also to avoid issues with IDE variable viewers)
        if isinstance(item, tuple):
            if (len(item) == self.ndim) and _np.array([isinstance(x, int) for x in item]).all():
                # print(1)
                return _ma.MaskedArray(data = selected_values,
                                       mask = selected_mask,
                                       fill_value = self._fill_value)
        
        #If we are selecting on 0 axis using a list/ndarray
        #then 
        #if selecting a continuous subset (diffs are always the same), fine
        #otherwise we lose the temporal dimension
        #and we should just return the selected masked array
        if (isinstance(item_0, list)) or (isinstance(item_0, _np.ndarray)):

            diffs = _np.diff(item)
            if len(_np.unique(diffs))>1:
                return _ma.MaskedArray(data = selected_values,
                                       mask = selected_mask,
                                       fill_value = _np.nan)


        #If selecting only one timepoint (=int on 0 axis)
        #maintain original number of dimensions
        elif isinstance(item_0, int):
            # print(3)
            selected_values = _ma.expand_dims(selected_values, 0)
            selected_mask = _ma.expand_dims(selected_mask, 0)

        elif item_0 is Ellipsis:
            pass
        
        elif item_0 is None:
            pass
        
        elif isinstance(item_0, slice):
            pass
        else:
            print('why here?')
            print(item_0, type(item_0))

        #######################
        # process temporal metadata
        #######################
        
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
        elif (isinstance(item_0, list)) or (isinstance(item_0, _np.ndarray)):
            start = item_0[0]
            new_start_time = start_time + start/sampling_frequency
            
            diffs = _np.unique(_np.diff(item_0))
            if len(diffs)==1:
                ratio = diffs[0]
            else:
                ratio = _np.nan
            new_sampling_freq = sampling_frequency/ratio

        elif isinstance(item_0, slice):
            start = item_0.start if item_0.start is not None else 0
            new_start_time = start_time + start/sampling_frequency
            
            ratio = item_0.step if item_0.step is not None else 1
            new_sampling_freq = sampling_frequency/ratio
        
        else: #slice
            print('why here?')
            print(item_0, type(item_0))
            new_start_time = _np.nan
            new_sampling_freq = _np.nan
            
        ######################
        # finalize
        # print(selected_values.shape)
        # print(selected_mask.shape)
        # print(new_sampling_freq, new_start_time)
        selected = self.__class__(selected_values, 
                                  new_sampling_freq,
                                  new_start_time,
                                  info,                                  
                                  mask = selected_mask)
        # print(type(selected))
        #process info
        selected = self.__getitem_attrib__(selected, item)
        # print(type(selected))
        
        return selected
    
    def __getitem_attrib__(self, selected, item):
        # print('--getitem_attrib--')
        # print(type(selected))
        # separate item for the first axis from others
        item_other = None
        
        if isinstance(item, tuple): #more than one axis involved
            item_0 = item[0]
            item_other = item[1:]
        else:
            item_0 = item
        
        #=========================
        # work on metadata in info dict
        info = selected.ph['info']
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
    
    def get_indices(self):
        selection = list(_np.zeros(self.ndim).astype(int))
        selection[0] = slice(None, None, None)
        selection = tuple(selection)
        # print(selection)
        mask_0 = self._mask.__getitem__(selection)
        indices = _np.where(~mask_0)[0]
        return indices
    
    def get_values(self):
        indices = self.get_indices()
        values = self.data[indices]
        return values
    
    @property
    def ph(self):
        return self._optinfo

    def __repr__(self):
        return self.get_values().__repr__() + '\n'+\
            f'{self.ph["sampling_freq"]} Hz \n'

import numpy as np

def info(s):
    print(type(s))
    print('shape\n', s.shape)
    print('ph \n', s.ph)
    print('values \n', s.get_values()[:2])
    
signal_values = np.random.uniform(size=(1000, 3, 5))
fsamp = 10

s = Signal(data = signal_values, 
           sampling_freq = fsamp)


#%%
mmm = ma.array([[1,2,3], [1,2,3]])
np.mean(mmm, axis=0)

np.mean(s, axis=0)

info(s)
info(s[100:])
info(s[100::2])

info(s[100, 0 ,0])

#%%
import numpy.ma as ma
import numpy as np

class NewMasked(ma.MaskedArray):
    def __new__(self, 
                data, 
                mask=False,
                info={}):
        
        obj = super(NewMasked, self).__new__(self, data=data, mask=mask).view(self)
        obj._optinfo = {'info': info}
        return obj
    
    def __array_finalize__(self, obj):
        print(type(obj))
        return super(NewMasked, self).__array_finalize__(obj)
    
m = NewMasked(np.random.uniform(size=(10,5,3)), info={1:1})    

r = np.mean(m, axis=0)
print(m._optinfo)
print(r._optinfo)

#%%
                      
## create an Unevenly signal defining the instants
x_values_time = np.arange(100)/fsamp
x_values_time[-1] = 12.5
x_values_time += 10

s_fake_time = Signal(values = signal_values,
                     sampling_freq = fsamp, 
                     start_time = tstart,
                     x_values = x_values_time, 
                     x_type = 'instants')

s_fake_time_evenly = s_fake_time.fill(kind = 'linear')
