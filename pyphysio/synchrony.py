#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:57:48 2024

@author: bizzego
"""
import numpy as _np
import statsmodels.api as _sm
from scipy.stats import median_abs_deviation as _median_abs_deviation

def _IRLS(y, X, max_iter=50):
    done = False
    iterations = 0
    beta_old = _np.ones(X.shape[1])
    #initialize weights to ones
    weights = _np.ones(len(y))
    while(not done):
        #a- solve beta by WLS
        #fit weighted LS
        model_WLS = _sm.WLS(y, X, weights=weights)
        results_WLS = model_WLS.fit()
        #get new beta
        beta_new = results_WLS.params
        
        #b- recalculate weights
        residuals_WLS = results_WLS.resid
        weights = _sm.robust.norms.TukeyBiweight(c=4.685).weights(residuals_WLS)
        change = abs(_np.min((beta_new - beta_old)/beta_old))
        
        #c- repeat steps 5a-b until changes in beta are small (<1%)
        if (change <0.01) or (iterations >= max_iter):
            done = True
        
        beta_old = beta_new
        
        iterations +=1
    return(beta_new)

def robust_correlation(signal_1, signal_2=None, channels=None):
    '''
    Santosa et al 2017 "Characterization and correction of the false-discovery rates in resting state connectivity using functional near-infrared spectroscopy"
    
    NOTE: signal_1 and signal_2 are assumed pre-whitened
    Supports multiple channels (all channels pairs will be considered)
    If multiple components, each component is considered separately
    
    if signal_2 is not provided, functional connectivity is computed
    if channels is not provided, all channels are considered
    '''
    
    #check dims
    values_1 = signal_1.p.get_values()
    
    if signal_2 is None:
        values_2 = values_1
        idx_offset = 1 #if no signal_2, then do not compute the correlation for same channels
    else:
        values_2 = signal_2.p.get_values()
    
        for i,j in zip(values_1.shape, values_2.shape):
            assert i == j, "Sizes are not the same"
        idx_offset = 0
    
    if channels is None:
        channels = _np.arange(values_1.shape[1])
    
    n_components = values_2.shape[2]
    corr_mat = _np.ones(shape=(len(channels), len(channels), n_components))
    
    for i_comp in range(n_components):
        for i_ch in _np.arange(len(channels)):
            ch_1 = channels[i_ch]
            y_1 = values_1[:,ch_1,i_comp]
            
            for j_ch in _np.arange(i_ch+idx_offset, len(channels)):
                ch_2 = channels[j_ch]
                y_2 = values_2[:,ch_2,i_comp]
                
                #% preweighting
                r = [_np.sqrt(x**2 + y**2) for (x, y) in zip(y_1, y_2)]
                sigma = 1.4826*_median_abs_deviation(r)
                r_norm = r/sigma
                
                weights = _sm.robust.norms.TukeyBiweight(c=4.685).weights(r_norm)
                
                y_1_s = y_1*weights
                y_2_s = y_2*weights
                
                X1 = _np.expand_dims(y_2_s, 1)
                X1 = _np.concatenate([_np.ones(shape=(len(y_2_s), 1)), X1],
                                     axis=1) #add constant term
                
                betaAB = _IRLS(y_1_s, X1)
                
                X2 = _np.expand_dims(y_1_s, 1)
                X2 = _np.concatenate([_np.ones(shape=(len(y_1_s), 1)), X2], 
                                     axis=1) #add constant term
                
                betaBA = _IRLS(y_2_s, X2)
                
                R = _np.sqrt(betaAB[1]*betaBA[1])
                
                corr_mat[i_ch, j_ch, i_comp] = R
                corr_mat[j_ch, i_ch, i_comp] = R

    
    return(corr_mat)