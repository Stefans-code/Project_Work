# coding=utf-8
import pyphysio as ph
import numpy as np
import numpy.ma as ma
np.random.seed(1234)

sizes = [1000, (1000, 5), (1000, 5, 3)]

def general_tests(s, f=None, t = None, v=None, i_sh=None ):
    assert s.get_sampling_freq() == f, s.get_sampling_freq()
    assert s.get_start_time() == t,s.get_start_time()
    assert s.get_info()['info'] == 'info',s.get_info()
    
    
    assert s.ndim == v.ndim, s.ndim
    s_shape = s.shape
    for i in range(len(s_shape)):
        assert s_shape[i] == v.shape[i]
    
    if v.ndim > 1:
        assert s.get_nchannels() == v.shape[1], s.get_nchannels()
        
    if v.dim > 2:
        assert s.get_ncomponents() == v.shape[2], s.get_ncomponents()

    
    s.set_start_time(12)
    assert s.get_start_time() == 12
    
    s.set_sampling_freq(12)
    assert s.get_sampling_freq() == 12

    indices = s.get_indices()
    values = s.get_values()
    
    v_shape = values.shape
    for i in range(len(v_shape)):
        assert v_shape[i] == indices.shape[i]
    
    assert s.get_end_time() == t + (len(v) + 1)/f
    
    assert s.IDX2idx(123) == s.get_indices()[123], s.IDX2idx(123)
    assert s.idx2IDX(0) == 0
    assert s.idx2IDX(1000) == len(s.get_indices()) - 1,  s.idx2IDX(1000) #CHECK
    
    assert isinstance(s.ph, dict)
    assert len(s.ph) >= 3

    # get_duration
    assert isinstance(s.get_duration(), float)
    assert self.s.get_duration() == s.get_end_time() - s.get_start_time()

    # plot
    import matplotlib.lines
    a = s.plot()
    assert isinstance(a, list)
    assert isinstance(a[0], matplotlib.lines.Line2D)
    import matplotlib.collections
    a = s.plot("ro")
    assert isinstance(a, list)
    assert isinstance(a[0], matplotlib.lines.Line2D)
    assert isinstance(s.plot("|r"), matplotlib.collections.LineCollection)

    # pickleability
    ps = ph.Signal.from_pickleable(s.pickleable)
    assert ps.get_start_time() == s.get_start_time()
    assert ps.get_sampling_freq() == s.get_sampling_freq()
    assert ps.get_end_time() == s.get_end_time()
    assert ps.get_signal_nature() == s.get_signal_nature()
    assert len(ps) == len(s)
    
    t = s.get_times()
    assert len(s.get_values()) == len(t)  # length
    assert len(np.where(np.diff(t) <= 0)[0]) == 0  # strong monotonicity

    # get_time_from_iidx
    assert s.idx2time(0) == s.get_time(0)
    assert s.idx2time(len(t) - 1) == s.get_times()[len(t) - 1]
    assert s.idx2time(len(t)) == s.get_times()[len(t)]
    
# noinspection PyAttributeOutsideInit
class TestSignal(object):
    def setup_class(self):
        self.info = {'info': 'info'}
        self.number12 = 12
        self.start_times = [-100, 0, 0.00001]
        self.fsamps = [10, 0.1, 10.1]
        self.fresamp = [10, 0.001, 100]

        random_values = [np.random.uniform(size = x) for x in [sizes]]
        ones_values = [np.ones(shape = x) for x in [sizes]]
        zeros_values = [np.zeros(shape = x) for x in [sizes]]

        self.values = [random_values, ones_values, zeros_values]
        self.x_indices = [np.arange(1000), 
                          np.arange(0,2000,2), 
                          np.unique(np.logspace(0, 5, 2000).astype(int))[:1000]]
        
        self.x_instants = [x/self.fsamp for x in self.x_indices]
        
        no_masks = np.zeros(1000)
        
        mask_01 = no_masks.copy()
        mask_01[::2] = 1
        mask_random = no_masks.copy()
        mask_random[np.random.permutation(1000)[:100]] = 1
        
        self.masks = [no_masks, mask_01, mask_random]
        

    def test_no_mask_signal(self):
        
        #test simple signal
        for f in self.fsamps:
            for t in self.start_times:
                for v in self.values:
                    for i_sh, sh in enumerate(sizes):
                        s = ph.Signal(v[i_sh], sampling_freq=f, 
                                      start_time=t, info=self.info)
                        assert s.get_end_time() == t + (len(v) + 1)/f, s.get_end_time()
                        assert s.get_indices()[123] == 123, s.get_indices()[123]
                        assert s.get_times()[123] == 123/f + t, s.get_times()[123]
                        
                        assert np.sum(s.get_values()) == np.sum(v)
                        
                        general_tests(s,f,t,v,i_sh)
                        
    def test_mask_signal(self):
        for i_m, m in enumerate(self.masks):
            for f in self.fsamps:
                for t in self.start_times:
                    for v in self.values:
                        for i_sh, sh in enumerate(self.shapes):
                            s = ph.Signal(v[i_sh], sampling_freq=f, 
                                          start_time=t, info=self.info, mask=m)
                        
                            start_time = s.get_start_time()
                            indices = s.get_indices()
                            
                            assert np.sum(s.get_values()) == np.sum(v[m])
                            
                            if i_m == 0:
                                assert len(indices) == 1000, len(indices)
                                assert start_time == t, start_time
                                assert s.get_times()[123] == 123/f + t
                            if i_m == 1:
                                assert len(indices) == 500, len(indices)
                                assert start_time == t + 1/f, start_time
                            if i_m == 2:
                                assert len(indices) == 100, len(indices)
                                true_t = np.where(indices==1)[0][0]*f
                                assert start_time == true_t, start_time
                            
                            general_tests(s,f,t,v,i_sh)
