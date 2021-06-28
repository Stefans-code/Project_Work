# coding=utf-8
import numpy as _np
import numpy.ma as _ma
from scipy import interpolate as _interp
from matplotlib.pyplot import ylabel as _ylabel, grid as _grid, subplots as _subplots,\
    tight_layout as _tight_layout, subplots_adjust as _subplots_adjust,\
        xlim as _xlim, gcf as _gcf, sca as _sca, gca as _gca

from numbers import Number as _Number
import copy

def from_pickleable(pickle):
    """
    Builds a Signal using the pickleable tuple version of it.
    :param pickle: Tuple of the form (Signal, ph dict).
    :return: Signal
    """
    d, ph, info = pickle
    assert isinstance(d, Signal)
    assert isinstance(ph, dict)
    assert isinstance(info, dict)
    d._optinfo = ph
    d._optinfo['info'] = info
    return d


def from_pickle(path):
    """
    Loads a Signal from a pickle file given the path.
    :param path: File system path to the pickle file.
    :return: A Signal.
    """
    from gzip import open
    from pickle import load
    f = open(path)
    p, ph, info_pickle = load(f)
    f.close()
    # return from_pickleable(p)
    p._optinfo = ph
    info = {}
    for info_key in info_pickle.keys():
        if info_key == 'sqi':
            info_sqi = {}
            for k in info_pickle['sqi'].keys():
                info_sqi[k] = from_pickleable(info_pickle['sqi'][k])
            info['sqi'] = info_sqi
        
        elif info_key == 'good':
            info['good'] = from_pickleable(info_pickle['good'])
        
        elif info_key == 'stim':
            info['stim'] = from_pickleable(info_pickle['stim'])
        else:
            info[info_key] = info_pickle[info_key]
    p._optinfo['info'] = info
    
    return p


class Signal(_ma.MaskedArray):
    """
    Signal
    
    Attributes:
    -----------
    
    data : numpy.array (TIME [,1 [,1]])
        Values of the signal
    sampling_freq : float, >0
        Sampling frequency, This also sets the precision for the temporal localization of the signal samples
    start_time: float,
        Instant of signal start
    info : dict, default = {}
        Other info
    mask : ndarray of bools
        The mask with False when the index corresponds to a sample
    
    x_values : numpy.array of int
        Instants, or indices when the values are measured.
    x_type : str
        Type of x values given.
        Can be 'indices' or 'instants'

    Information from sampling_freq, x_values and x_type is converted into a mask
    instants when the mask is False indicate the temporal position of the signal values
    on an EvenlySignal with the given sampling frequency and start time.
    
    idx: indices of the EvenlySignal
    IDX: indices of the non masked samples[0, 189, 213, 454,  ...]
    IDX is the result of self.get_indices()
    
    idx*fsamp gives the instants
    idx[IDX]*fsamp is the timestamp of the IDX-th value of the signal
    
    When slicing, indices of the first axis are considered idxs.
    
    """
    def __new__(self, data, mask=None,
                #>>>this is required for correct pickling, dont know why
                dtype=None,
                copy=False, 
                subok=True,
                ndmin=0,
                fill_value=None,
                keep_mask=True,
                hard_mask=None,
                shrink=True,
                order=None,                
                #<<<this is required for correct pickling, dont know why
                sampling_freq=_np.nan, 
                start_time = 0, 
                info={}, 
                x_values = None, 
                x_type='indices'):
        
        #TODO assert max dims = 3 ?
        
        if mask is not None:
            #TODO we should check that mask is uniform across dimensions
            #TODO: check that shapes are compatible
            
            if mask.sum() != 0: #we are defining UE using mask, x_values should be none
                assert x_values is None, "When defining masked values, x_values should be none"
            
            new_data = data
            
        elif x_values is not None: #we are defining UE using x_values
            assert x_type in ['indices', 'instants'], "x_type not in ['indices', 'instants']"
        
            assert len(x_values) == len(data), "Length mismatch (y:%d vs. x:%d)" % (len(data), len(x_values))
            assert len(_np.where(_np.diff(x_values) <= 0)[0]) == 0, 'Given x_values are not strictly monotonic'
        
            if x_type == 'instants':
                assert start_time<= x_values[0], 'the first instant is before the start_time'
            
            #we avoid initial masked values,
            #by updating the start_time and x_values
            if x_type == 'indices':
                start_time = start_time + x_values[0]/sampling_freq
                x_values = x_values - x_values[0]
                size_0 = x_values[-1] + 1
            else:
                start_time = x_values[0]
                size_0 = int(_np.ceil(sampling_freq*(x_values[-1] - start_time))) + 1
        
            #compute non masked indices
            if x_type == 'indices':
                idx_0 = x_values
            else:
                idx_0 = _np.round(sampling_freq*(x_values - start_time)).astype(int)
            
            #creating output shape
            values_shape = list(data.shape)
            new_shape = values_shape
            new_shape[0] = size_0
            new_shape = tuple(new_shape)
            
            #create data of the Signal
            new_data = _np.empty(new_shape)
            new_data[idx_0] = data
            
            #create mask
            mask = _np.ones(new_shape)
            mask[idx_0] = 0
            mask = mask.astype(bool)
            
        else:
            new_data = data
            mask = _np.zeros_like(new_data).astype(bool)

        obj = _ma.MaskedArray(data=new_data, mask=mask).view(self)
        
        obj._optinfo = {'sampling_freq': sampling_freq,
                        'start_time': start_time,
                        'info': info}
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        mask = _np.zeros_like(len(self)).astype(bool)
        self._mask = getattr(obj, '_mask', mask)
        self._hardmask = getattr(obj, '_hardmask', False)
        self._fill_value = getattr(obj, '_fill_value', _np.nan)
        
        if hasattr(obj, '_optinfo'):
            self._optinfo = getattr(obj, '_optinfo')
        else:
            _optinfo = {'sampling_freq':_np.nan,
                        'start_time':_np.nan,
                        'info': {}}
            self._optinfo = _optinfo
            # print('no _optinfo')
    
    def clone_properties(self, new_values, new_mask=None, x_values=None,x_type=None):
        x_new = Signal(new_values, new_mask,
                       sampling_freq=self.get_sampling_freq(),
                       start_time=self.get_start_time(),
                       info=self.get_info(),
                       x_values=x_values,
                       x_type=x_type)
        return(x_new)
        
    def __getitem__(self, item):
        # print(item)
        sampling_frequency = copy.deepcopy(self.get_sampling_freq())
        start_time = copy.deepcopy(self.get_start_time())
        info = copy.deepcopy(self.get_info())

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
        selected = self.__class__(selected_values, mask = selected_mask,
                                  sampling_freq=new_sampling_freq,
                                  start_time=new_start_time,
                                  info=info)
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
                    sqi = info['sqi'][s]
                    new_sqi[s] = sqi.__getitem__(item_new)
                    
                selected.update_info('sqi', new_sqi)
                
            if 'good' in info.keys():
                # new_good = {}
                # for s in info['good'].keys():
                good = info['good']
                new_good = good.__getitem__(item_new)
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
    
    def get_times(self):
        indices = self.get_indices()
        return indices / self.get_sampling_freq() + self.get_start_time()

    def get_end_time(self):
        return self.get_start_time() + (1+self.get_indices()[-1])/self.get_sampling_freq()
    
    def idx2time(self, idx):
        return idx / self.get_sampling_freq() + self.get_start_time() if idx is not None else None

    def time2idx(self, time):
        idx = int((time - self.get_start_time()) * self.get_sampling_freq())
        if idx < 0:
            idx=0
        return(idx)

    def idx2IDX(self, idx):
        '''
        Return the nearest non masked index before the given idx 
        
        Parameters
        ----------
        idx : int
            Target index

        Returns
        -------
        idx. the nearest index of the 0 axis to IDX

        '''
        idx_t = self.get_indices()
        #the index corresponding to IDX should not be after the target idx
        IDX = _np.where((idx_t - idx)<=0)[0][-1]
        return(IDX)
    

    def IDX2idx(self, IDX):
        '''
        Return the IDX-th non masked index
        
        Parameters
        ----------
        IDX : int
            Index of the non masked index to be returned.

        Returns
        -------
        idx. The IDX-th non masked index
        '''

        idx_t = self.get_indices()
        idx = idx_t[IDX]
        return(idx)
    
    def IDX2time(self, IDX):
        idx = self.IDX2idx(IDX)
        time = super().idx2time(idx)
        return time
    
    def time2IDX(self, time):
        idx = super().time2idx(time)
        IDX = self.idx2IDX(idx)
        return IDX
    
    def segment_time(self, t_start, t_stop=None):
        """
        Segment the signal given a time interval

        Parameters
        ----------
        t_start : float
            The instant of the start of the interval
        t_stop : float 
            The instant of the end of the interval. By default is the end of the signal

        Returns
        -------
        portion : UnvenlySignal
            The selected portion
        """
        idx_start = self.time2idx(t_start)
        idx_stop = self.time2idx(t_stop)
        return self[idx_start:idx_stop]
    
    def segment_IDX(self, IDX_start, IDX_stop=None):
        """
        Segment the signal using the indices of the non masked indices

        Parameters
        ----------
        IDX_start : int
            index of the non masked index to start
        IDX_stop : float
            index of the non masked index to stop

        Returns
        -------
        portion : UnvenlySignal
            The selected portion
        """
        idx_start = self.IDX2idx(IDX_start)
        idx_stop = self.IDX2idx(IDX_stop)
        
        return self[idx_start:idx_stop]
    
    @property
    def ph(self):
        return self._optinfo

    def clone(self):
        #TODO CHECK
        print('using Signal.clone(); might not work properly')
        return copy.deepcopy(self)
        

    def has_multi_channels(self):
        return(self.get_nchannels()>1)
    
    def get_nchannels(self):
        if self.ndim>1:
            return(self.shape[1])
        else:
            return(1)
    
    def has_multi_components(self):
        return(self.get_ncomponents()>1)
    
    def get_ncomponents(self):
        if self.ndim>2:
            return(self.shape[2])
        else:
            return(1)
    
    def is_onedim(self):
        if self.ndim == 1:
            return True
        if self.has_multi_channels():
            return False
        if self.has_multi_components():
            return False
        return True
    
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

    def set_info(self, info):
        self.ph['info'] = info
    
    def has_good(self, good_global=True):
        if 'good' in self.ph['info'].keys():
            if not good_global:
                return True
            else:
                good = self.ph['info']['good']
                if good.shape[0] == 1:
                    return True
                return False
        return False
    
    def get_good(self, safe=True):
        assert self.has_good(), "Quality has not been computed yet"
        
        info = self.get_info()
        is_good = info['good']
        # assert is_good.shape[0] == 1, "Quality has not been computed globally. Please compute global quality first"
        
        if is_good.ndim == 1:
            return(_np.array(_np.where(is_good))[0])
        else:
            return(_np.array(_np.where(is_good)[1:]))
    
    def update_info(self, key, value):
        self.ph['info'][key] = value
    
    def get_duration(self):
        return self.get_end_time() - self.get_start_time()

    def is_masked(self):
        return(self.mask.sum()>0)
    
    def fill(self, kind='cubic'):
        """
        Fill the masked values using interpolation
        
        Parameters
        ----------
        kind : str
            Method for interpolation: 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'

        Returns
        -------
        signal: Signal
            The signal without masked values
        """
        
        if not self.is_masked():
            return self.clone()
        
        #add a sample at the end to avoid errors
        values = self.get_values()
        values = _np.append(values, values[[-1]], axis=0)
        indices = self.get_indices()
        indices = _np.append(indices, indices[[-1]]+1, axis=0)
        
        assert kind != 'cubic' or len(values) > 3, "At least 4 samples needed for cubic interpolation"

        tck = _interp.interp1d(indices, values, kind=kind, axis=0)

        # Exclusive end, same x_value
        indices_out = _np.arange(self.data.shape[0])
        # print(indices_out)
        sig_out = tck(indices_out)

        # Init new signal
        sig_out = Signal(data=sig_out, mask=None,
                         sampling_freq=self.get_sampling_freq(),
                         start_time=self.get_start_time(),
                         info=self.get_info())

        return sig_out
    
    def resample(self, fout, kind='linear'):
        """
        Resample a signal

        Parameters
        ----------
        fout : float
            The sampling frequency for resampling
        kind : str
            Method for interpolation: 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'

        Returns
        -------
        resampled_signal : Signal
            The resampled signal
        """
        
        if self.is_masked():
            self_interp = self.fill()
        else:
            self_interp = self.clone()
            
        ratio = self_interp.get_sampling_freq() / fout

        if fout < self_interp.get_sampling_freq() and ratio.is_integer():  # fast downsampling
            return self_interp[::int(ratio)]
        
        else:
            #add a sample at the end to avoid errors
            values = self_interp.get_values()
            values = _np.append(values, values[[-1]], axis=0)
            indices = self_interp.get_indices()
            indices = _np.append(indices, indices[[-1]]+1, axis=0)
            indices_out = _np.arange(self_interp.shape[0] * fout / self_interp.get_sampling_freq()) * ratio
            
            print(indices.shape)
            print(values.shape)
            print(indices[-1], indices_out[-1])
            tck = _interp.interp1d(indices, values, kind=kind, axis=0)
            signal_out = tck(indices_out)

        return Signal(signal_out,mask=None,
                      sampling_freq=fout,
                      start_time=self_interp.get_start_time(),
                      info=self_interp.get_info())

    def plot(self, marker=None, ncols=4, sharey=True):
        fig = _gcf()
        
        #if single signal, then plot
        if self.is_onedim():
            
            #TODO if existing figure has many axes, 
            #replicate the plot on each axis
            
            #if good then use asolid line
            #else use a dotted line
            linestyle='solid'
            if self.has_good():
                good = self.get_good()
                if len(good)==0:
                    linestyle = 'dotted'
            
            #plot the signal
            ax = _gca()
            t_ = self.get_times()
            v_ = self.get_values()
            
            if marker is None:
                if self.is_masked(): 
                    marker = '.'
            
            if marker is None:                
                ax.plot(t_, _np.squeeze(v_), linestyle = linestyle)
            else:
                ax.plot(t_, _np.squeeze(v_), marker, linestyle = linestyle)
            _grid(True)
        
        else:
            
            n_ch = self.get_nchannels()
            n_comp = self.get_ncomponents()

            #if existing figure has enough number of axes
            #use the figure
            if len(fig.axes)>= n_ch:
                axes = fig.axes
            
            #else create a new figure 
            else: 
                if n_ch>1:
                    #compute number of cols and rows and create a new figure
                    n_cols = n_ch if n_ch < ncols else ncols
                    n_rows = int(_np.ceil(n_ch/n_cols))
                    
                    fig, axes = _subplots(n_rows, n_cols, num = fig.number, sharex=True, sharey=sharey)
                    axes = axes.ravel()
                else:
                    fig, axes = _subplots(1, 1, num = fig.number, sharex=True, sharey=sharey)
                    axes = [axes]
            
            #recursive calls to signal.plot()
            #for each channel and component
            for i_ch in range(n_ch):
                _sca(axes[i_ch])
                
                if n_comp>1:
                    for i_comp in range(n_comp):
                        self[:, i_ch, i_comp].plot(marker=marker)
                else:
                    self[:, i_ch].plot(marker=marker)
                _ylabel(i_ch)
                _grid(True)
                
            _xlim(self.get_start_time(), self.get_end_time())
            _tight_layout()
            _subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, hspace=0.2, wspace=0.2)
        
    @property
    def pickleable(self):
        """
        Returns a pickleable tuple of this Signal.
        :return: Tuple (Signal, ph dict).
        """
        info = self.get_info()
        
        
        info_pickle = {}
        for info_key in info.keys():
            if info_key == 'sqi':
                info_sqi = {}
                for k in info['sqi'].keys():
                    info_sqi[k] = info['sqi'][k].pickleable
                info_pickle['sqi'] = info_sqi
            
            elif info_key == 'good':
                if hasattr(info['good'], 'pickleable'):
                    info_pickle['good'] = info['good'].pickleable
            
            elif info_key == 'stim':
                if hasattr(info['stim'], 'pickleable'):
                    info_pickle['stim'] = info['stim'].pickleable
            else:
                info_pickle[info_key] = info[info_key]
                
        return self, self._optinfo, info_pickle

    def to_pickle(self, path):
        """
        Saves this Signal into a pickle file.
        :param path: File system path to the file to write (create/overwrite).
        """
        from gzip import open
        from pickle import dump
        f = open(path, "wb")
        dump(self.pickleable, f, protocol=2)
        f.close()
       
    def __repr__(self):
        return self.get_values().__repr__() + '\n'+\
            f'{self.get_sampling_freq()} Hz \n'+\
                f'{self.get_start_time()} s \n'

#%%
# import numpy as np

# signal_values = np.arange(100)
# fsamp = 10
# tstart = 0 

# ## create an Unevenly signal defining the instants
# x_values_time = np.arange(100)/fsamp
# x_values_time[-1] = 12.5
# x_values_time += 10

# s_fake_time = Signal(values = signal_values, 
#                      sampling_freq = fsamp, 
#                      start_time = tstart,
#                      x_values = x_values_time, 
#                      x_type = 'instants')

# s_fake_time_evenly = s_fake_time.fill(kind = 'linear')
