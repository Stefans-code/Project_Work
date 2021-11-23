import xarray as xr




ds = xr.open_dataset("~/Downloads/sst.mnmean.nc", engine="netcdf4")

#%%
t = ds["sst"].data

lat_points = xr.DataArray([60, 80, 90], dims="points")
lon_points = xr.DataArray([250, 250, 250], dims="points")

#%%
import numpy as np
import xarray as xr

rng = np.random.default_rng(seed=0)  # we'll use this later

da = xr.DataArray(
    np.ones((3, 4, 2)),
    dims=("x", "y", "z"),
    name="a",
    coords={"z": [-1, 1], "u": ("x", [0.1, 1.2, 2.3])},
    attrs={"attr": "value"},
)

#%%
da = xr.DataArray(
    np.ones((3, 4)),
    dims=("x", "y"),
    coords={
        "x": ["a", "b", "c"],
        "y": np.arange(4),
        "u": ("x", np.arange(3), {"attr1": 0}),
    },
)

ds = xr.Dataset(
    data_vars={
        "a": (("x", "y"), np.ones((3, 4))),
        "b": ("t", np.full((8,), 3), {"attr": "value"}),
    },
    coords={
        "x": [-1, 0, 1],
    },
    attrs={"attr": "value"},
)

#%%
import pandas as pd

fsamp = 128

values = np.random.uniform(size = (5, 2, 1000))
t_start = 23.145

signal = xr.DataArray(
    values,
    dims = ('channel', 'component', 'time'),
    coords = {'time': pd.timedelta_range(start=0, periods = values.shape[2], freq=f'{10**9/fsamp}ns'),
              'channel': np.arange(values.shape[0]),
              'component': np.arange(values.shape[1]),
              'cluster1': ("channel", [0,2,0,1,1], {"type": 1}),
              'cluster2': ("channel", [0,0,1,1,1])},
    name = 'signal')

print(signal)