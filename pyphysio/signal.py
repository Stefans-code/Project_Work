# coding=utf-8
import numpy as _np
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
    d, ph = pickle
    assert isinstance(d, Signal)
    assert isinstance(ph, dict)
    d._pyphysio = ph
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
    p = load(f)
    f.close()
    return from_pickleable(p)


class Signal(_np.ndarray):
    
    def __new__(cls, values, sampling_freq, start_time=None, info = {}):
        assert sampling_freq > 0, "The sampling frequency cannot be zero or negative"
        assert start_time is None or isinstance(start_time, _Number), "Start time is not numeric"
        
        #convert values to ndarray
        values = _np.asarray(values)
        
        # R2
        # # a "simple" signal
        # # will have at least one channel and one component
        # if values.ndim == 1:
        #     values = _np.expand_dims(values, 1)
        # if values.ndim == 2:
        #     values = _np.expand_dims(values, 2)
        
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
        
    def __getitem__(self, item):
        # print(item, self.shape)
        #TODO if float segment based on time

        #apply __getitem__ to values (ndarray)
        values = self.get_values()
        selected_values = values.__getitem__(item)
        
        #If we are selecting an index on all axis 
        #then we are extracting only one scalar:
        #just return the scalar and avoid processing the attributes
        #(This is also to avoid issues with IDE variable viewers)
        if isinstance(item, tuple):
            if _np.array([isinstance(x, int) for x in item]).all():
                return selected_values
            
        #try to catch Ellipsis issue with the np.apply_along_axis
        #which uses ellipsis and changes the shape
        #avoid processing attributes and just return the signal
        if isinstance(item, tuple) and Ellipsis in item:
            return(self.clone_properties(selected_values))
        
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
        # selected = selected.clone()
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
        
    @property
    def ph(self):
        return self._pyphysio

    def clone(self):
        obj = self.copy()
        obj._pyphysio = copy.deepcopy(self.ph)
        return(obj)
    
    def get_values(self):
        return _np.asarray(self)

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

    def set_info(self, value):
        self.ph['info'] = value    
    
    def has_good(self):
        # info = self.get_info()
        #TODO return if good are global?
        return 'good' in self.ph['info'].keys()
    
    def get_good(self):
        assert self.has_good(), "Quality has not been computed yet"
        info = self.get_info()
        is_good = info['good']
        assert is_good.shape[0] == 1, "Quality has not been computed globally. Please compute global quality first"
        
        if is_good.ndim == 1:
            return(_np.array(_np.where(is_good))[0])
        else:
            return(_np.array(_np.where(is_good)[1:]))
    
    def update_info(self, key, value):
        self.ph['info'][key] = value
    
    def get_duration(self):
        return self.get_end_time() - self.get_start_time()
    
    def time2idx(self, time):
        idx = int((time - self.get_start_time()) * self.get_sampling_freq())
        if idx < 0:
            idx=0
        return(idx)
    
    def idx2time(self, idx):
        time = self.get_start_time() + idx/self.get_sampling_freq()
        return(time)
    
    def clone_properties(self, new_values):
        # x_new = self.__class__(new_values,
        #                        self.get_sampling_freq(),
        #                        self.get_start_time(),
        #                        self.get_info())
        # return(x_new)
        pass
    
    # @_abstract
    def get_times(self):
        pass
    
    # @_abstract
    def get_end_time(self):
        pass

    # @_abstract
    def resample(self, fout, kind='linear'):
        pass

    # @_abstract
    def segment_time(self, t_start, t_stop=None):
        pass

    def plot(self, marker=None, ncols=4):
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
                    
                    fig, axes = _subplots(n_rows, n_cols, num = fig.number, sharex=True)
                    axes = axes.ravel()
                else:
                    fig, axes = _subplots(1, 1, num = fig.number, sharex=True)
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
        return self, self.ph

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
        return f"<start_time: {self.get_start_time()}>"

    #TODO: implement __array_ufunc__ ?

class EvenlySignal(Signal):
    """
    Evenly spaced signal
    
    Attributes:
    -----------
    
    data : numpy.array, (TIME [, CHANNELS [, COMPONENTS]])
        Values of the signal
    sampling_freq : float, >0
        Sampling frequency
    start_time: float,
        Instant of signal start
    info : dict, default = {}
        Other info 
    """
    
    def get_times(self):
        return _np.arange(self.shape[0]) / self.get_sampling_freq() + self.get_start_time()

    def get_end_time(self):
        return self.get_time(self.shape[0] - 1) + 1. / self.get_sampling_freq()

    def get_time(self, idx):
        return idx / self.get_sampling_freq() + self.get_start_time() if idx is not None else None

    def get_value_t(self, instant):
        values = self.get_values()
        idx = self.time2idx(instant)
        return values[idx]
    
    def clone_properties(self, new_values):
        x_new = self.__class__(new_values,
                               self.get_sampling_freq(),
                               self.get_start_time(),
                               self.get_info())
        return(x_new)
    
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
        resampled_signal : EvenlySignal
            The resampled signal
        """

        ratio = self.get_sampling_freq() / fout

        if fout < self.get_sampling_freq() and ratio.is_integer():  # fast interpolation
            signal_out = self.get_values()[::int(ratio)]
        else:
            # The last sample is doubled to allow the new size to be correct
            indexes = _np.arange(self.shape[0] + 1)
            indexes_out = _np.arange(self.shape[0] * fout / self.get_sampling_freq()) * ratio
            self_l = _np.append(self, _np.expand_dims(self[-1], 0), axis=0)

            tck = _interp.interp1d(indexes, self_l, kind=kind, axis=0)
            signal_out = tck(indexes_out)

        return EvenlySignal(values=signal_out,
                            sampling_freq=fout,
                            start_time=self.get_start_time(),
                            info=self.get_info())

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
        portion : EvenlySignal
            The selected portion
        """

        return self[self.time2idx(t_start): self.time2idx(t_stop)]
    
    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " freq:" + str(self.get_sampling_freq()) + "Hz>\n" + self.view(
            _np.ndarray).__repr__()


class UnevenlySignal(Signal):
    """
    Unevenly spaced signal
    
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
    
    
    x_values : numpy.array of int
        Instants, or indices when the values are measured.
    x_type : str
        Type of x values given.
        Can be 'indices' or 'instants'

    duration: float,
        Duration of the original EvenlySignal, if any. Duration is needed to have information about the duration of the
        last sample, if None the last sample will last 1. / fsamp.
    
    Information from x_values and x_type is converted into a signal attribute: idx_t
    idx_t is the array with the indices that indicate the temporal position of the signal values
    on an EvenlySignal with the given sampling frequency and start time.
        
    IDX: indices of the idx_t attribute [0, 1, 2,   3,  4,  5, ...]
    idx: values of the idx_t attribute  [1, 3, 10, 13, 20, 55, ...]
    
    idx*fsamp give the instants
    idx[IDX]*fsamp is the timestamp of the IDX-th value of the signal
    
    When slicing, indices of the first axis are considered IDXs.
    
    """

    def __new__(cls, values, sampling_freq=1000, start_time=None, info={}, 
                x_values=None, x_type='instants'):
        
        #TODO: if indices, x_values should always start at 0
        #check that UEvenly signals are created accordingly
        obj = Signal.__new__(cls, values=values,
                             sampling_freq=sampling_freq,
                             start_time=start_time,
                             info=info)

        assert x_values is not None, "x_values are missing"
        assert x_type in ['indices', 'instants'], "x_type not in ['indices', 'instants']"
        
        x_values = _np.array(x_values).ravel()
        assert len(x_values) == len(values), "Length mismatch (y:%d vs. x:%d)" % (len(values), len(x_values))
        assert len(_np.where(_np.diff(x_values) <= 0)[0]) == 0, 'Given x_values are not strictly monotonic'
            
        #manage idx_t
        #and consistent start_time
        if x_type == 'indices':
            # Keep indices, set start_time
            if start_time is None:
                start_time = 0
            assert _np.array([isinstance(x, _np.integer) for x in x_values]).all(), "x_values should be integers when x_type is indices"
            idx_t = _np.array(x_values).astype(int)
            
        else: #x_type is instants
            assert start_time is None, "start_time should be None when x_values are instants"
            start_time = x_values[0]
            
            # WARN: limitation to 10 decimals due to workaround to prevent wrong cast flooring
            # (e.g. np.floor(0.29 * 100) == 28)
            idx_t = _np.round((x_values - start_time) * sampling_freq, 10).astype(int)

        obj.ph['start_time'] = start_time
        obj.ph['idx_t'] = idx_t
        return obj

    #!!!
    def __getitem__(self, item):
        #first, apply __getitem__ to values (ndarray)
        values = self.get_values()
        selected_values = values.__getitem__(item)
        
        #If we are selecting an index on all axis 
        #then we are extracting only one scalar:
        #just return the scalar and avoid processing the attributes
        #(This is also to avoid issues with IDE variable viewers)
        if isinstance(item, tuple):
            if _np.array([isinstance(x, int) for x in item]).all():
                return selected_values
        
        #ISSUE 1
        #There are some numpy functions (eg median)
        #that fuck the shape of the signal
        #and would generate errors with the slicing of the idx_t
        #catch these issues by comparing the len of values and idx_t
        #and just return a numpy.ndarray object
        idx_t = self.get_indices()
        n_samples = self.shape[0]
        if len(idx_t) != n_samples:
            print('issue1')
            print(item)
            print(len(idx_t))
            print(idx_t)
            print(n_samples, self.shape)
            print(self.shape)
            return(selected_values)
        
        #We already processed the values before,        
        #here we process the idx_t attribute
        #considering only the selection on the 0 axis
        idx_t = self.get_indices()
        item_0 = item[0] if isinstance(item, tuple) else item
        new_idx_t = idx_t.__getitem__(item_0)
        
        if isinstance(new_idx_t, _Number):
            offset_time = new_idx_t/self.get_sampling_freq()
            new_idx_t = _np.array([0])
        else:
            offset_time = new_idx_t[0]/self.get_sampling_freq()
            new_idx_t = new_idx_t - new_idx_t[0]
        
        #ISSUE 2
        #try to catch Ellipsis issue with some np functions
        #which uses ellipsis and changes the shape
        if isinstance(item, tuple) and Ellipsis in item:
            if len(selected_values.shape) == 0:
                return(self.clone_properties(_np.array([selected_values]),
                                             _np.array([new_idx_t[0]]),
                                             'indices'))
            elif len(idx_t) == len(selected_values):
                return(self.clone_properties(selected_values,
                                             new_idx_t,
                                             'indices'))
            else:
                assert len(selected_values) == len(new_idx_t), "why here?"
                return(selected_values)
                
        
        #to understand when it is not working:
        assert len(selected_values) == len(new_idx_t)
        
        selected = self.clone_properties(selected_values,
                                         new_idx_t,
                                         'indices')
        
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
        
        #manage start time which was changed in __getitem_attrib__
        selected.set_start_time(self.get_start_time() + offset_time)
        return(selected)
        
    def __getitem_attrib__(self, item, selected):
        return super().__getitem_attrib__(item, selected)
        
    def clone_properties(self, new_values, new_x, new_x_type):
        assert new_x is not None
        assert new_x_type is not None
        assert new_values.shape[0] == new_x.shape[0],\
                "new_values and new_x shold have the same length"
        
        x_new = UnevenlySignal(new_values,
                               self.get_sampling_freq(),
                               self.get_start_time(),
                               self.get_info(),
                               new_x,
                               new_x_type)
        return(x_new)

    def get_end_time(self):
        return self.get_start_time() + (1+self.get_indices()[-1])/self.get_sampling_freq()

    def get_times(self):
        return self.ph['idx_t'] / self.get_sampling_freq() + self.get_start_time()

    def get_indices(self):
        return self.ph['idx_t']

    def IDX2time(self, IDX):
        idx = self.IDX2idx(IDX)
        time = super().idx2time(idx)
        return time
    
    def time2IDX(self, time):
        idx = super().time2idx(time)
        IDX = self.idx2IDX(idx)
        return IDX
    
    
    def idx2IDX(self, idx):
        '''
        Return the nearest index of the 0 axis
        to the IDX-th index of the idx_t attibute

        Parameters
        ----------
        IDX : int
            Index of the idx_t.

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
        Return the idx-th value of the signal indices (idx_t attribute)    
        
        Parameters
        ----------
        idx : int
            Index of the idx_t to be returned.

        Returns
        -------
        IDX. The idx-th value of the signal indices
        '''

        idx_t = self.get_indices()
        idx = idx_t[IDX]
        return(idx)

    def to_evenly(self, kind='cubic'):
        """
        Interpolate the UnevenlySignal to obtain an evenly spaced signal
        Parameters
        ----------
        kind : str
            Method for interpolation: 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'

        Returns
        -------
        interpolated_signal: ndarray
            The interpolated signal
        """

        assert kind != 'cubic' or len(self) > 3, "At least 4 samples needed for cubic interpolation"

        data_x = self.ph['idx_t']  # From a constant freq range
        data_y = self.get_values()

        # Cubic if needed
        if kind == 'cubic':
            tck = _interp.InterpolatedUnivariateSpline(data_x, data_y, axis=0)
        else:
            tck = _interp.interp1d(data_x, data_y, kind=kind, axis=0)

        # Exclusive end, same x_value
        x_out = _np.arange(data_x[0], data_x[-1] + 1)
        sig_out = tck(x_out)

        # Init new signal
        sig_out = EvenlySignal(values=sig_out,
                               sampling_freq=self.get_sampling_freq(),
                               start_time=self.get_start_time(),
                               info=self.get_info())

        return sig_out

    def resample(self, fout, kind='linear'):
        return self.to_evenly(kind).resample(fout, kind)

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
        IDX_start = self.idx2IDX(idx_start)
        
        idx_stop = self.time2idx(t_stop)
        IDX_stop = self.idx2IDX(idx_stop)
        return self[IDX_start:IDX_stop]

    def segment_idx(self, idx_start, idx_stop=None):
        """
        Segment the signal using the indices of the idx_t attribute

        Parameters
        ----------
        idx_start : int
            The index of the start of the interval
        idx_stop : float
            The index of the end of the interval. By default is the end of the signal

        Returns
        -------
        portion : UnvenlySignal
            The selected portion
        """
        IDX_start = self.idx2IDX(idx_start)
        IDX_stop = self.idx2IDX(idx_stop)
        
        return self[IDX_start:IDX_stop]
    
    def plot(self, marker="."):
        super().plot(marker = marker)

        
    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " time resolution:" + str(1 / self.get_sampling_freq()) + "s>\n" + \
               self.get_values().__repr__() + " Times\n:" + self.get_times().__repr__()