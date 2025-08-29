"""
This script fits 2- and 3-layer EBMs to various lengths of ESM output data,
and saves the resultant parameters.

Author: Chris Wells - August 2025
"""
import glob
import xarray
import numpy as np
import pickle
import os
import energy_balance_model as ebm

alpha = 5E-4

layers = [2, 3] 

lengths = [150, 300, 500, 900, 1000, 1800, 2000, 3000, 4000, 5000, 5900]

esm_files = glob.glob('../data/rtmt_tas_anom/tas*nc')
esms = []
for esm_file in esm_files:
    esm = esm_file.split("tas_glbmean_")[1].split("_abrupt4x")[0]
    if esm not in esms:
        esms.append(esm)
        
cmip6_esm_files = glob.glob('../data/rtmt_tas_anom/CMIP6_models/tas*nc')
cmip6_esms = []
for cmip6_esm_file in cmip6_esm_files:
    cmip6_esm = cmip6_esm_file.split("tas_glbmean_")[1].split("_abrupt4x")[0]
    if cmip6_esm not in cmip6_esms:
        cmip6_esms.append(cmip6_esm)
                
full_esms = esms + cmip6_esms


if os.path.isfile('../data/output/ebm_params.pkl'):
    with open('../data/output/ebm_params.pkl', 'rb') as handle:
        ebm_params = pickle.load(handle)
else:
    ebm_params = {}
    
for esm in full_esms:
    print(esm)
    if esm not in ebm_params.keys():
        ebm_params[esm] = {}
    
    if esm in esms:
        tas_file_in = glob.glob(f'../data/rtmt_tas_anom/tas_*_{esm}_*nc')[0]
        rtmt_file_in = glob.glob(f'../data/rtmt_tas_anom/rtmt_*_{esm}_*nc')[0]
    else:
        tas_file_in = glob.glob(f'../data/rtmt_tas_anom/CMIP6_models/tas_*_{esm}_*nc')[0]
        rtmt_file_in = glob.glob(f'../data/rtmt_tas_anom/CMIP6_models/rtmt_*_{esm}_*nc')[0]

    ds_tas = xarray.open_dataset(tas_file_in, decode_times=False)
    
    if esm == 'NorESM2-LM':
        tas = ds_tas.tas.values
    else:
        tas = ds_tas.tas_glbmean.values
    n_yrs = len(tas)
    print(f'# years: {n_yrs}')
    
    ds_rtmt = xarray.open_dataset(rtmt_file_in, decode_times=False)
    
    if esm == 'NorESM2-LM':
        rtmt = ds_rtmt.rtmt.values
    else:
        rtmt = ds_rtmt.rtmt_glbmean.values

    y = np.stack((tas, rtmt), axis=1)
    
    lengths_full = lengths + [n_yrs]
    
    for l in lengths_full:
        if l <= n_yrs:
            for k in layers:
                if f'{k}layer' not in ebm_params[esm].keys():
                    ebm_params[esm][f'{k}layer'] = {}
                if l not in ebm_params[esm][f'{k}layer'].keys():
                    print(f'Do {l} years for {k} layer')
                    ebm_params[esm][f'{k}layer'][l] = {}
                    
                    try:
                        model, res = ebm.fit_model(y[:l,:], k=k, alpha=alpha)
                    except:
                        print(f'failed for {k} layer, L={l}, {esm}')
                        continue
                    ebm_params[esm][f'{k}layer'][l]["C"] = model.C
                    ebm_params[esm][f'{k}layer'][l]["kappa"] = model.κ
                    ebm_params[esm][f'{k}layer'][l]["epsilon"] = model.ϵ
                    ebm_params[esm][f'{k}layer'][l]["F4xCO2"] = model.F
                                        
                    with open('../data/output/ebm_params.pkl', 'wb') as handle:
                        pickle.dump(ebm_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    
#%%

with open('../data/output/ebm_params.pkl', 'wb') as handle:
    pickle.dump(ebm_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
