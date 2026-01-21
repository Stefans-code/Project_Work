#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 15:19:51 2026

@author: bizzego
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Generate Synthetic Data ---
np.random.seed(42)
n = 1000
# Create two series that are highly correlated initially
data1 = np.random.normal(0, 1, n)
data2 = data1 * 0.9 + np.random.normal(0, 0.5, n)  # Correlation ~0.9

#%%
# Flip the relationship halfway through (Regime Change)
split = n // 2
data2[split:] = data1[split:] * -0.9 + np.random.normal(0, 0.5, n - split)

# Put into a DataFrame
df = pd.DataFrame({'Series_A': data1, 'Series_B': data2})

# --- 2. Compute Correlations ---

# Method A: Standard Rolling Window (Simple Moving Average)
# Window = 60 days. Note: This will react slowly to the change.
rolling_corr = df['Series_A'].rolling(window=20).corr(df['Series_B'])

# Method B: Exponentially Weighted Moving Correlation (EWMC)
# Span = 60. Note: This gives more weight to recent data.
ewm_corr = df['Series_A'].ewm(span=20).corr(df['Series_B'])

#%%
# --- 3. Visualization ---
plt.figure(figsize=(12, 6))

# Plot the calculated correlations
plt.plot(rolling_corr, label='Rolling Window (Simple)', linestyle='--', color='gray', alpha=0.7)
plt.plot(ewm_corr, label='EWMC (Exponential)', color='blue', linewidth=2)

# Add a vertical line where the correlation actually flipped
plt.axvline(x=split, color='red', linestyle=':', label='True Correlation Flip')

plt.title('Comparison: Rolling Window vs. EWMC')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Time Step')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


#%%
import xarray as xr
import numpy as np
import pandas as pd # Used only to generate sample time index

# --- 1. Setup Sample Xarray Data ---
# Creating two time series with a shared 'time' dimension
times = pd.date_range("2023-01-01", periods=100)
ds = xr.Dataset(
    {
        "sig1": (("time",), np.random.randn(100)),
        "sig2": (("time",), np.random.randn(100)),
    },
    coords={"time": times},
)

da1 = ds["sig1"]
da2 = ds["sig2"]

da1, da2 = xr.align(da1, da2, join="inner")
    
ewm = da1.rolling_exp(time=10, window_type='span')

    # Calculate EWM Means
    # We need a fresh rolling_exp object for each calculation to ensure safety
    # Helper to get EWM mean easily:
    def get_ewm_mean(data):
        return data.rolling_exp(time=span, window_type='span').mean()

    # 1. EWM Means of the series
    mean_x = get_ewm_mean(da1)
    mean_y = get_ewm_mean(da2)

    # 2. EWM Covariance
    # Cov(X, Y) = E[(X - mu_x)(Y - mu_y)]
    # A numerically stable approximation for EWM is: E[XY] - E[X]E[Y]
    # However, strictly for EWM, it is safer to smooth the cross-products of demeaned data
    # or use the identity: EWM((x - mean_x)*(y - mean_y))
    
    # Approach: Calculate (x - mean_x) * (y - mean_y) at every step
    # Note: This is an approximation. For exact recursive EWM covariance, 
    # pandas uses a specific recursive formulation. 
    # For vectorized xarray, smoothing the product of deviations is the standard approach.
    
    cov_xy = get_ewm_mean((da1 - mean_x) * (da2 - mean_y))
    var_x = get_ewm_mean((da1 - mean_x)**2)
    var_y = get_ewm_mean((da2 - mean_y)**2)

    # 3. Correlation
    corr = cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))
    
    return corr