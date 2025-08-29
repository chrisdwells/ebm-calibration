"""
This script processes NorESM gridded data to create the annual inputs used
for this model's calibration. It doesn't need to be run here to reproduce 
the results, as the output is already in data/output.

Author: Chris Wells - August 2025
"""

import xarray
import numpy as np

datadir = '../data/NorESM2-LM/'

varlist = ['tas', 'rtmt']
for var in varlist:

    ds_pi = xarray.open_mfdataset(f'{datadir}/{var}*_piControl_*.nc')

    var_pi = ds_pi[var]
    var_pi = var_pi.resample(time="YE").mean()

    weights = np.cos(np.deg2rad(var_pi.lat))
    weights.name = "weights"

    var_pi_weighted = var_pi.weighted(weights)
    var_pi_weighted_mean = var_pi_weighted.mean(("lon", "lat"))

    pi_offset = var_pi_weighted_mean.mean().values


    ds = xarray.open_mfdataset(f'{datadir}/{var}*_abrupt-4xCO2*.nc')

    var = ds[var]
    var = var.resample(time="YE").mean()

    weights = np.cos(np.deg2rad(var.lat))
    weights.name = "weights"

    var_weighted = var.weighted(weights)
    var_weighted_mean = var_weighted.mean(("lon", "lat"))
    
    var_offset = var_weighted_mean - pi_offset
    var_offset.to_netcdf(f'../data/rtmt_tas_anom/{var}_glbmean_NorESM2-LM_abrupt4x_0001-0500_anom.nc')
