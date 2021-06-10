#%%
import numpy as np

#%%
class C(np.ndarray):
    def __new__(cls, *args, **kwargs):
        print('In __new__ with class %s' % cls)
        return super(C, cls).__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        # in practice you probably will not need or want an __init__
        # method for your subclass
        print('In __init__ with class %s' % self.__class__)

    def __array_finalize__(self, obj):
        print('In array_finalize:')
        print('   self type is %s' % type(self))
        print('   obj type is %s' % type(obj))

#%%
# Explicit constructor
c = C((10,))

# View casting
a = np.arange(10)
cast_a = a.view(C)

# Slicing (example of new-from-template)
cv = c[:1]

#%%


class InfoArray(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, info=None):
        # Create the ndarray instance of our type, given the usual
        # ndarray input arguments.  This will call the standard
        # ndarray constructor, but return an object of our type.
        # It also triggers a call to InfoArray.__array_finalize__
        obj = super(InfoArray, subtype).__new__(subtype, shape, dtype,
                                                buffer, offset, strides,
                                                order)
        # set the new 'info' attribute to the value passed
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.info = getattr(obj, 'info', None)
        # We do not need to return anything

#%%
import numpy as np
import numpy.ma as ma

class RealisticInfoArray(ma.MaskedArray):

    def __new__(cls, input_array, mask=False, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = ma.MaskedArray(input_array, mask=mask, hard_mask=False).view(cls)
        print(obj._mask)
        print(obj._fill_value)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        super(RealisticInfoArray, self).__array_finalize__(obj)
        self.info = getattr(obj, 'info', None)
        self._mask = getattr(obj, '_mask', None)
        self._fill_value = getattr(obj, '_fill_value', None)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        print('ufunc')
        return super(RealisticInfoArray, self).__array_ufunc__(ufunc, method,
                                                               *args, **kwargs)
        
    def __array_wrap__(self, out_arr, context=None):
        print('wrap')
        print(context)
        if isinstance(out_arr, RealisticInfoArray):
            obj = ma.MaskedArray.__array_wrap__(self, out_arr, context)
            self.info = getattr(obj, 'info', None)
            self._mask = getattr(obj, '_mask', None)
            self._fill_value = getattr(obj, '_fill_value', None)
        else:
            return out_arr

#%%
arr = np.arange(5)

obj = RealisticInfoArray(arr, info='information')
type(obj)
obj.info

#%%
v = obj[1:]
type(v)
v.info

v3 = obj+3
v3.info


#%% ++++++++++++++++++++++++++++++++++++++++
import numpy as np
import numpy.ma as ma

class RealisticInfoArray(np.ndarray):
    def __new__(cls, input_array, mask=False, info=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        obj.mask = mask
        obj.info = info
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)
        self.mask = getattr(obj, 'mask', None)

    # def __array_ufunc__(self, ufunc, method, *args, **kwargs):
    #     print('ufunc')
    #     return super(RealisticInfoArray, self).__array_ufunc__(ufunc, method,
    #                                                            *args, **kwargs)
        
    # def __array_wrap__(self, out_arr, context=None):
    #     print('wrap')
    #     print(context)
    #     if isinstance(out_arr, RealisticInfoArray):
    #         obj = ma.MaskedArray.__array_wrap__(self, out_arr, context)
    #         self.info = getattr(obj, 'info', None)
    #         self._mask = getattr(obj, '_mask', None)
    #         self._fill_value = getattr(obj, '_fill_value', None)
    #     else:
    #         return out_arr

#%%
arr = np.arange(5)

obj = RealisticInfoArray(arr, info='information')
type(obj)
obj.info

#%%
v = obj[1:]
type(v)
v.info

v3 = obj+3
v3.info

#%%

class A(np.ndarray):
    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        info = getattr(self, 'info', None)
        results = super(A, self).__array_ufunc__(ufunc, method, *args, **kwargs)
        
        if results is NotImplemented:
            return NotImplemented

        results.info = info

        return results


#%%
a = np.arange(5.).view(A)

b = np.sin(a)

b.info
#%%
import pyphysio as ph
import numpy as np

signal = ph.Signal(np.random.uniform(size=1000), 10)

#signal = ph.Signal(np.random.uniform(size=(1000,3,5)), 10)
print(type(signal))
print(signal._pyphysio)

#%%
signal3 = signal+3
print(type(signal3))
print(signal3._pyphysio)

#%%
from pynirs.loaders import load_nirx

DATADIR = '/home/bizzego/UniTn/data/fnirs_sexism/original/2021-02-05_001'

signal = load_nirx(DATADIR, False)

#%%
type(signal)

type(signal+3)

type(signal*3)

type(signal/np.nanmean(signal))
