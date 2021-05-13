# coding=utf-8
import numpy as _np
from scipy import interpolate as _interp
from matplotlib.pyplot import ylabel as _ylabel, grid as _grid, subplots as _subplots,\
    tight_layout as _tight_layout, subplots_adjust as _subplots_adjust,\
        xlim as _xlim, gcf as _gcf, sca as _sca, gca as _gca

from numbers import Number as _Number
import copy

# TODO: indexing should split also infos (eg sqi, good)
# !!!: standard array indexing (signal[idx_start:idx_stop])

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
                   
        obj = _np.asarray(values).view(cls)
        
        obj._pyphysio = {
            'sampling_freq': sampling_freq,
            'start_time': start_time if start_time is not None else 0,
            'info': info
        }
        
        setattr(obj, "_mutated", False)
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
        selected_values = super().__getitem__(item)
        selected_info = self.__getiteminfo__(self, item)
        selected_values.set_info(selected_info)
        return selected_values
        
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
    
    def get_channel(self, idx_ch):
        assert self.ndim > 1, "Signal has not multiple channels"
        assert self.get_nchannels() > idx_ch, f"Index of the channel {idx_ch}; Signal has {self.get_nchannels()} channels"
        ch_values = self.get_values()[:, idx_ch]
        #recover original number of dimensions
        ch_values = _np.expand_dims(ch_values, 1)
        return(EvenlySignal(ch_values, self.get_sampling_freq(), self.get_start_time(), self.get_info()))
    
    def get_component(self, idx_comp):
        assert self.ndim > 2, "Signal has not multiple components"
        assert self.get_ncomponents() > idx_comp, f"Index of the component {idx_comp}; Signal has {self.get_ncomponents()} components"
        comp_values = self.get_values()[:, :, idx_comp]
        #recover original number of dimensions
        comp_values = _np.expand_dims(comp_values, 2)
        return(EvenlySignal(comp_values, self.get_sampling_freq(), self.get_start_time(), self.get_info()))
    
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
        info = self.get_info()
        return 'good' in info.keys()
    
    def get_good(self):
        info = self.get_info()
        assert 'good' in info.keys(), "Quality has not been computed yet"
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
    
    def get_idx(self, time):
        idx = int((time - self.get_start_time()) * self.get_sampling_freq())
        if idx < 0:
            idx=0
        return(idx)
        
    # @_abstract
    def clone_properties(self):
        pass
    
    def __getiteminfo__(self, item):
        pass
    
    # @_abstract
    def get_times(self):
        pass
    
    # @_abstract
    def get_end_time(self):
        pass

    # @_abstract
    def get_iidx(self, time):
        pass

    # @_abstract
    def get_time(self, idx):
        pass

    # @_abstract
    def get_time_from_iidx(self, iidx):
        pass

    # @_abstract
    def resample(self, fout, kind='linear'):
        pass

    # # @_abstract
    # def segment_idx(self, idx_start, idx_stop=None):
    #     pass

    # # @_abstract
    # def segment_iidx(self, iidx_start, iidx_stop=None):
    #     pass

    # @_abstract
    def segment_time(self, t_start, t_stop=None):
        pass

    def plot(self, style="", ncols=4):
        fig = _gcf()
        
        ndims = self.ndim
        
        if ndims ==  1:
            if self.has_good():
                good = self.get_good()
                
                print(good.shape)
                if len(good)>0:
                    linestyle = '-'
                else:
                    linestyle = '--'
            ax = _gca()
            t_ = self.get_times()
            ax.plot(t_, self, style, linestyle = linestyle)
            _grid(True)
        
        else:
            n_ch = self.get_nchannels()
            n_comp = self.get_ncomponents()
            
            if len(fig.axes)>= n_ch:
                    axes = fig.axes
            else:
                if n_ch>1:
                    
                    n_cols = n_ch if n_ch < ncols else ncols
                    n_rows = int(_np.ceil(n_ch/n_cols))
                    
                    fig, axes = _subplots(n_rows, n_cols, num = fig.number, sharex=True)
                    axes = axes.ravel()
                else:
                    fig, axes = _subplots(1, 1, num = fig.number, sharex=True)
                    axes = [axes]
        
            for i_ch in range(n_ch):
                _sca(axes[i_ch])
                
                if n_comp>1:
                    for i_comp in range(n_comp):
                        self[:, i_ch, i_comp].plot()
                else:
                    self[:, i_ch].plot()
                _ylabel(i_ch)
                _grid(True)
                
            _xlim(self.get_start_time(), self.get_end_time())
            _tight_layout()
            _subplots_adjust(top=0.9, bottom=0.01, left=0.05, right=0.95, hspace=0.2, wspace=0.2)
        

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

    #TODO: implement __array_ufunc__

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
    
    def clone_properties(self, new_values):
        x_new = EvenlySignal(new_values,
                             self.get_sampling_freq(),
                             self.get_start_time(),
                             self.get_info())
        return(x_new)

    def get_times(self):
        return _np.arange(self.shape[0]) / self.get_sampling_freq() + self.get_start_time()

    def get_end_time(self):
        return self.get_time(self.shape[0] - 1) + 1. / self.get_sampling_freq()

    def get_iidx(self, time):
        return self.get_idx(time)

    def get_time(self, idx):
        return idx / self.get_sampling_freq() + self.get_start_time() if idx is not None else None

    def get_time_from_iidx(self, iidx):
        return self.get_time(iidx)

    def get_value_t(self, instant):
        values = self.get_values()
        nearest_idx = int(_np.round(self.get_sampling_freq() * (instant - self.get_start_time())))
        assert nearest_idx < self.shape[0], "Required instant is after the end of the signal"  # return self[-1]
        assert nearest_idx >= 0, "Required instant is before the start of the signal"  # return self[0]
        
        return values[nearest_idx]
    
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

    # def segment_idx(self, idx_start, idx_stop=None):
    #     """
    #     Segment the signal given the indexes

    #     Parameters
    #     ----------
    #     idx_start : int or None
    #         The index of the start of the interval
    #     idx_stop : int or None
    #         The index of the end of the interval. By default is the length of the signal

    #     Returns
    #     -------
    #     portion : EvenlySignal
    #         The selected portion
    #     """
    #     return self.segment_iidx(idx_start, idx_stop)

    def segment_iidx(self, iidx_start, iidx_stop=None):

        signal_values = self.get_values()

        if iidx_start is None:
            iidx_start = 0
        if iidx_stop is None:
            iidx_stop = len(self)

        values = signal_values[int(iidx_start):int(iidx_stop)]

        out_signal = self.clone_properties(values)
        out_signal.set_start_time(self.get_time(iidx_start))
        return out_signal

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

        return self.segment_idx(self.get_idx(t_start), self.get_idx(t_stop))
    
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
        Sampling frequency
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
    """

    def __new__(cls, values, sampling_freq=1000, start_time=None, info={}, 
                x_values=None, x_type='instants'):
        
        assert x_values is not None, "x_values are missing"
        assert x_type in ['indices', 'instants'], "x_type not in ['indices', 'instants']"
        x_values = _np.asarray(x_values)
        assert len(x_values) == len(values), "Length mismatch (y:%d vs. x:%d)" % (len(values), len(x_values))
        assert len(_np.where(_np.diff(x_values) <= 0)[0]) == 0, 'Given x_values are not strictly monotonic'

        assert values.shape[0] == x_values.shape[0], "Length of x_values should be equal to the length of the values"
        
        if x_type == 'indices':
            # Keep indices, set start_time
            if start_time is None:
                start_time = 0
            idx_t = x_values
        else: #x_type ss instants
            
            # Get indices removing start_time
            if start_time is None:
                start_time = x_values[0]
            else:
                assert start_time <= x_values[0], "More than one sample at or before start_time"
            
            # WARN: limitation to 10 decimals due to workaround to prevent wrong cast flooring
            # (e.g. np.floor(0.29 * 100) == 28)
            idx_t = _np.round((x_values - start_time) * sampling_freq, 10).astype(int)

        obj = Signal.__new__(cls, values=values,
                             sampling_freq=sampling_freq,
                             start_time=start_time,
                             info=info)

        obj.ph['idx_t'] = idx_t
        return obj


    #TODO: test
    def clone_properties(self, new_values, new_x=None, new_x_type=None):
        if new_x is not None:
            assert new_values.shape[0] == new_x.shape[0],\
                "new_values and new_x shold have the same length"
        else:
            new_x = self.ph['x_values']
            new_x_type = 'indices'
            
        x_new = UnevenlySignal(new_values,
                               self.get_sampling_freq(),
                               self.get_start_time(),
                               self.get_info(),
                               new_x,
                               new_x_type)
        return(x_new)

    def get_duration(self):
        return self.ph['duration']

    def get_end_time(self):
        return self.get_start_time() + self.get_duration()

    def get_times(self):
        return self.ph['x_values'] / self.get_sampling_freq() + self.get_start_time()

    def get_indices(self):
        return self.ph['x_values']

    def get_time(self, idx):
        return idx / self.get_sampling_freq() + self.get_start_time() if idx is not None else None

    def get_time_from_iidx(self, iidx):
        if len(self) == 0:
            return self.get_start_time()
        elif int(iidx) < len(self):
            return self.get_indices()[int(iidx)] / self.get_sampling_freq() + self.get_start_time()
        else:
            return self.get_time_from_iidx(-1)

    def get_iidx(self, time):
        return self.get_iidx_from_idx((time - self.get_start_time()) * self.get_sampling_freq())

    def get_iidx_from_idx(self, idx):
        if idx >= self.get_indices()[0]:
            return int(_np.searchsorted(self.get_indices(), idx))
        else:
            return None

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

        data_x = self.ph['x_values']  # From a constant freq range
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
                               start_time=self.get_time_from_iidx(0),
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

        return self.segment_idx(self.get_idx(t_start) if t_start is not None else None,
                                self.get_idx(t_stop) if t_stop is not None else None)

    def segment_idx(self, idx_start, idx_stop=None):
        """
        Segment the signal given the indexes

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
        if idx_start is None:
            idx_start = 0
        if idx_stop is None:
            idx_stop = self.get_indices()[-1]

        iib = self.get_iidx_from_idx(idx_start)
        iie = self.get_iidx_from_idx(idx_stop)

        if iib is None and iie is None:
            iidx_start = iidx_stop = idx_start = idx_stop = 0
        else:
            iidx_start = int(iib) if iib is not None else 0
            iidx_stop = int(iie) if iie is not None else -1

        return UnevenlySignal(values=self.get_values()[iidx_start:iidx_stop],
                              sampling_freq=self.get_sampling_freq(),
                              start_time=self.get_time(idx_start),
                              info=self.get_info(),
                              x_values=self.get_indices()[iidx_start:iidx_stop] - idx_start,
                              x_type='indices',
                              duration=(idx_stop - idx_start) / self.get_sampling_freq())

    def segment_iidx(self, iidx_start, iidx_stop=None):
        """
        Segment the signal given the inner indexes

        Parameters
        ----------
        iidx_start : int
            The index of the start of the interval
        iidx_stop : float
            The index of the end of the interval. By default is the end of the signal

        Returns
        -------
        portion : UnvenlySignal
            The selected portion
        """
        if iidx_stop is None:
            iidx_stop = len(self)
        if iidx_start is None:
            iidx_start = 0
        if iidx_stop < len(self):
            idx_stop = self.get_indices()[int(iidx_stop)]
        else:
            idx_stop = self.get_indices()[-1] + 1
        idx_start = self.get_indices()[int(iidx_start)]

        return UnevenlySignal(values=self.get_values()[int(iidx_start):int(iidx_stop)],
                              sampling_freq=self.get_sampling_freq(),
                              start_time=self.get_time_from_iidx(iidx_start),
                              info=self.get_info(),
                              x_values=self.get_indices()[int(iidx_start):int(iidx_stop)] - self.get_indices()[int(iidx_start)],
                              x_type='indices',
                              duration=(idx_stop - idx_start) / self.get_sampling_freq())

    def plot(self, style=".-"):
        super().plot(style = style)

        
    def __repr__(self):
        return Signal.__repr__(self)[:-1] + " time resolution:" + str(1 / self.get_sampling_freq()) + "s>\n" + \
               self.get_values().__repr__() + " Times\n:" + self.get_times().__repr__()