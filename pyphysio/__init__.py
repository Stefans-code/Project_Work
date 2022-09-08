import numpy as _np
import xarray as _xr

_xr.set_options(keep_attrs=True)

try:
    from dask import __name__ as _
    scheduler = 'threads'
    # available schedulers:
    # #distributed, multiprocessing, processes, single-threaded, sync, synchronous, threading, threads
    print('Using dask. Scheduler: threads')
except:
    scheduler = 'single-threaded'
    
print("Please cite:")
print("Bizzego et al. (2019) 'pyphysio: A physiological signal processing library for data science approaches in physiology', SoftwareX")


#namespace
from .signal import *


