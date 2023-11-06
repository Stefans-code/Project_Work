# coding=utf-8
# from __future__ import print_function
# from __future__ import division

import matplotlib.pyplot as plt
import numpy as _np
from . import create_signal

class _MouseSelectionFilter(object):
    def __init__(self, onselect):
        self._select = onselect
        self._last_press = None

    def on_move(self, event):
        self._last_press = None

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        self._last_press = x, y, event.button

    def on_release(self, event):
        x, y = event.xdata, event.ydata
        if self._last_press is not None:
            xx, yy, b = self._last_press
            if x == xx and y == yy and event.button == b:
                self._select(event)


class _ItemManager(object):
    def __init__(self, snap_func, select, unselect, add, delete):
        self._snap_func = snap_func
        self._select = select
        self._unselect = unselect
        # self._delete = delete
        self._add = add
        self.selection = -1

    def unselect(self):
        self._unselect(self.selection)
        self.selection = None

    def on_select(self, ev):
        if ev.xdata is not None and ev.ydata is not None:
            x, y, item, new = self._snap_func(ev.xdata, ev.ydata)
#            print("on_select: %d, %d: %d" % (x, y, item))
            if self.selection is not None:
                self.unselect()
            if ev.button == 1:
                if new:
                    self._add(x, y, item)
                else:
                    self.selection = item
                    self._select(item)


class Annotate(object):
    
    def recompute_ibi(self):
        self.v_ibi = _np.repeat(_np.nan, len(self.v_ibi))
        t_ibi = self.t_ibi[self.idx_beats]
        v_ibi = _np.diff(t_ibi)        
        v_ibi = _np.insert(v_ibi, 0, v_ibi[0])
        self.idx_beats_good = [i for i in self.idx_beats if i not in self.idx_outliers]
        self.v_ibi[self.idx_beats] = v_ibi
    
    
    def __init__(self, ecg, ibi):
        # self.cursor = None
        
        self.beats = None
        self.outliers = None
        self.ibi_plot = None
        self.done = False
        
        self.ecg = ecg
        self.ibi = ibi
        
        self.t_ibi = ibi.p.get_times()
        self.t0 = self.t_ibi[0]
        self.v_ibi = ibi.p.get_values().ravel()
        self.idx_beats = _np.where(~_np.isnan(self.v_ibi))[0]
        self.idx_outliers = _np.array([self.idx_beats[0]])
        self.idx_beats_good = [i for i in self.idx_beats if i not in self.idx_outliers]
        
        self.min = _np.min(ecg.p.get_values())
        self.max = _np.max(ecg.p.get_values())
        
        self.fig, self.axes = plt.subplots(2,1, sharex=True)
        
        plt.sca(self.axes[0])
        ecg.p.plot()
        # ibi.p.plot('|')
        plt.sca(self.axes[1])
        ibi.p.plot('.')
        # ibi.p.plot('|')
        
        self.replot()

        class Cursor(object):
            left = None
            right = None
            radius = .3
            radiusi = int(radius * self.ecg.p.get_sampling_freq())

            @staticmethod
            def on_move(event):
                Cursor.draw(event)

            @staticmethod
            def on_scroll(event):
                if event.button == "up":
                    Cursor.radiusi += 3
                elif event.button == "down":
                    Cursor.radiusi -= 7
                Cursor.radius = Cursor.radiusi / self.ecg.p.get_sampling_freq()
                Cursor.draw(event)

            @staticmethod
            def draw(event):
                if Cursor.left is not None:
                    Cursor.left.remove()
                    Cursor.right.remove()
                    Cursor.left = None
                    Cursor.right = None
        
                if event.xdata is not None:  # TODO (Andrea): not do this if speed (dxdata/dt) is high
                    Cursor.left = self.axes[0].vlines(event.xdata - Cursor.radius, 
                                                      self.min,
                                                      self.max, 'k')
                    
                    Cursor.right = self.axes[0].vlines(event.xdata + Cursor.radius, 
                                                       self.min,
                                                       self.max, 'k')
                self.fig.canvas.draw()

        def find_peak(s):
            return _np.argmax(s)
       
        def snap(xdata, ydata):
            t_ibi = self.t_ibi[self.idx_beats]
            
            nearest_after = t_ibi.searchsorted(xdata)
            nearest_prev = nearest_after - 1

            dist_after = t_ibi[nearest_after] - xdata if 0 <= nearest_after < len(t_ibi) else None
            dist_prev = xdata - t_ibi[nearest_prev] if 0 <= nearest_prev < len(t_ibi) else None

            if dist_after is None or dist_prev < dist_after:
                if dist_prev is not None and dist_prev < Cursor.radius:
                    return t_ibi[nearest_prev], ydata, nearest_prev, False
            elif dist_prev is None or dist_after < dist_prev:
                if dist_after is not None and dist_after < Cursor.radius:
                    return t_ibi[nearest_after], ydata, nearest_after, False

            s = self.ecg.p.segment_time(xdata - Cursor.radius, xdata + Cursor.radius).p.get_values().ravel()
            s = _np.array(s)
            m = find_peak(s)
            return xdata - Cursor.radius + m / self.ecg.p.get_sampling_freq(), ydata, nearest_after, True

        class Selector(object):
            selector = None

            @staticmethod
            def select(item):
#                print("select: %d" % item)
                t_ibi = self.t_ibi[self.idx_beats]
                Selector.selector = self.axes[0].vlines(t_ibi[item], 
                                                        self.min, 
                                                        self.max, 
                                                        'g')

            @staticmethod
            def unselect(item):
                if Selector.selector is not None:
#                    print("unselect: %d" % item)
                    Selector.selector.remove()

        # it is correct that the computation of the values is done at the end (line 186)
        def add(time, y, pos):
            fsamp = self.ecg.p.get_sampling_freq()
            
            self.idx_beats = _np.insert(self.idx_beats, pos, (time - self.t0)*fsamp)
            self.replot()

        def delete(item):
            self.idx_beats = _np.delete(self.idx_beats, item)
            self.replot()
            
        def switch_outlier(item):
            idx_ibi_to_outlier = self.idx_beats[item]
            if idx_ibi_to_outlier not in self.idx_outliers:
                self.idx_outliers = _np.append(self.idx_outliers, idx_ibi_to_outlier)
            else:
                self.idx_outliers = [i for i in self.idx_outliers if i != idx_ibi_to_outlier]
            self.replot()
            
        
        def press(ev):
#            print(ev.key)
            if ev.key == "d" and im.selection is not None:
                delete(im.selection)
                im.unselect()
            
            if ev.key == "o" and im.selection is not None:
                switch_outlier(im.selection)
                im.unselect()
                
        def handle_close(ev):
            self.done = True
            return

        im = _ItemManager(snap, Selector.select, Selector.unselect, add, delete)
        mf = _MouseSelectionFilter(im.on_select)
            
        self.fig.canvas.mpl_connect('motion_notify_event', lambda e: (mf.on_move(e), Cursor.on_move(e)))
        self.fig.canvas.mpl_connect('button_press_event', mf.on_press)
        self.fig.canvas.mpl_connect('button_release_event', mf.on_release)
        self.fig.canvas.mpl_connect('scroll_event', Cursor.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', press)
        self.fig.canvas.mpl_connect('close_event', handle_close)
        
        while not self.done :
            plt.pause(1)
        
        plt.close(self.fig)

        
        self.recompute_ibi()
        
        self.v_ibi[self.idx_outliers] = _np.nan
        ibi_ok = create_signal(self.v_ibi, 
                               times=self.t_ibi, 
                               info = self.ibi.p.get_info())
        ibi_ok = ibi_ok.p.process_na('remove', na_remaining='remove')
        self.ibi_ok =  ibi_ok
        
    def __call__(self):
        return self.ibi_ok
    
    def replot(self):
        
        xlims = self.axes[0].get_xlim()
        self.recompute_ibi()
        
        if self.beats is not None:
            self.beats.remove()
        self.beats = self.axes[0].vlines(self.t_ibi[self.idx_beats], self.min, self.max, 'y')
        
        if self.outliers is not None:
            self.outliers.remove()
        self.outliers = self.axes[0].vlines(self.t_ibi[self.idx_outliers], self.min, self.max, 'r')
        
        if self.ibi_plot is not None:
            self.ibi_plot.remove()
        self.ibi_plot = self.axes[1].plot(self.t_ibi[self.idx_beats_good], 
                                          self.v_ibi[self.idx_beats_good], '.-', color='b')[0]
        
        self.axes[0].set_xlim(xlims)
        self.fig.canvas.draw()
