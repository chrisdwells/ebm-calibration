"""
This script uses the EBM parameters to run back through 4xCO2 experiments,
to study the quality of the fits against the original ESM data. It makes plots
S1 and S3 in the paper.

Author: Chris Wells - August 2025
"""
from fair import FAIR
from fair.interface import fill, initialise
import pickle
import glob
import xarray
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

layers = [2,3] 

with open('../data/output/ebm_params.pkl', 'rb') as handle:
    ebm_params = pickle.load(handle)

fair_4xco2_output = {}

esms = list(ebm_params.keys())
esms.sort()

l_list = []
for esm in ebm_params.keys():
    fair_4xco2_output[esm] = {}
    
    tas_file_in = glob.glob(f'../data/rtmt_tas_anom/tas_*_{esm}_*nc')[0]
        
    ds_tas = xarray.open_dataset(tas_file_in, decode_times=False)
    
    if esm == 'NorESM2-LM':
        tas = ds_tas.tas.values
    else:
        tas = ds_tas.tas_glbmean.values
    n_yrs = len(tas)
    
    fair_4xco2_output[esm]['ESM'] = tas
    
    for k in layers:
        fair_4xco2_output[esm][f'{k}layer'] = {}
        for l in ebm_params[esm][f'{k}layer'].keys():
            if len(ebm_params[esm][f'{k}layer'][l].keys()) == 0:
                print(f'no parameters for {esm} {k}layer {l}')
                continue

            l_list.append(l)
    
            f = FAIR(n_layers=k)
            f.define_time(0, n_yrs, 1)
            f.timebounds
            f.define_scenarios(['abrupt-4xCO2'])
            
            configs = [f'{esm}_{l}']
            f.define_configs(configs)
    
            species = ['CO2']
    
            properties = {
            'CO2': {
            'type': 'co2',
            'input_mode': 'forcing',
            'greenhouse_gas': True,
            'aerosol_chemistry_from_emissions': False,
            'aerosol_chemistry_from_concentration': False,
                    },
                }
            
            f.define_species(species, properties)
            f.allocate()
    
            initialise(f.temperature, 0)
            
            fill(f.climate_configs['gamma_autocorrelation'], 1000, config=configs[0])
            fill(f.climate_configs['stochastic_run'], False, config=configs[0])

            fill(f.climate_configs['ocean_heat_capacity'], ebm_params[esm][f'{k}layer'][l]["C"], config=configs[0])
            fill(f.climate_configs['ocean_heat_transfer'], ebm_params[esm][f'{k}layer'][l]["kappa"], config=configs[0])
            fill(f.climate_configs['deep_ocean_efficacy'], ebm_params[esm][f'{k}layer'][l]["epsilon"], config=configs[0])
    
            fill(f.forcing, ebm_params[esm][f'{k}layer'][l]["F4xCO2"], config=configs[0], specie='CO2')
                
            f.fill_species_configs()
            
            fill(f.species_configs['tropospheric_adjustment'], 0, specie='CO2')
            
            f.run()
            
            fair_4xco2_output[esm][f'{k}layer'][l] = f.temperature.loc[dict(layer=0, scenario='abrupt-4xCO2')][:-1,0]
            
#%%

# Fig S1 - 4xCO2 fits

cmap = mpl.colormaps['plasma']

lengths = {150:cmap(2/10),
           300:cmap(4/10), 
           900:cmap(6/10),
           }

fig, axs = plt.subplots(5, 4, figsize=(12, 15))
axs = axs.ravel()

for i, esm in enumerate(esms):

    l0 = fair_4xco2_output[esm]['ESM'].shape[0]
    time = np.arange(l0)
    
    axs[i].plot(time, fair_4xco2_output[esm]['ESM'], color='black', 
                    label = 'ESM')
    
    for l in lengths.keys():
        
        if l not in ebm_params[esm]['3layer'].keys():
            continue
        elif len(ebm_params[esm]['3layer'][l].keys()) == 0:
            continue
        
        axs[i].plot(time, fair_4xco2_output[esm]['3layer'][l], color=lengths[l],
                    label = f'{l} 3-layer')
        
    lmax = np.amax(np.asarray(list(ebm_params[esm]['3layer'].keys())))

    if lmax not in lengths.keys():
            
        if lmax in fair_4xco2_output[esm]['3layer'].keys():
                
            axs[i].plot(time, fair_4xco2_output[esm]['3layer'][lmax], color=cmap(8/10),
                        label = f'{lmax} 3-layer')
        
    axs[i].plot(time, fair_4xco2_output[esm]['2layer'][150], color='C2',
                label = '150 2-layer')
    
    axs[i].set_title(f'{esm}')
    axs[i].set_xlabel('Years')
    axs[i].set_ylabel('K')
    axs[i].legend()


for j in np.arange(len(ebm_params.keys()), 20):
    axs[j].set_visible(False)
    

plt.tight_layout()
plt.savefig(
    "../figures/for_paper/supplement/S1_4xCO2_timeseries.png"
)

#%%
# Fig S3 - RMSE 4xCO2

from matplotlib.lines import Line2D


layer_attrs = {
   '3layer':[['C0', 'o'], ['C1', 'x']],
   '2layer':[['C2', 's'], ['C3', '*']],
    }

fig=plt.figure(figsize=(8, 8))
ax=fig.add_subplot(111, label="1")

ax2 = ax.twinx()

rmse_errs = {}
rmse_errs['2layer'] = {}
rmse_errs['3layer'] = {}
rmse_errs['2layer']['Full'] = []
rmse_errs['3layer']['Full'] = []
rmse_errs['2layer']['Last100'] = []
rmse_errs['3layer']['Last100'] = []

for i, esm in enumerate(ebm_params.keys()):
    
    l0 = fair_4xco2_output[esm]['ESM'].shape[0]
    
    for l_i, l in enumerate(lengths.keys()):
        
        for k in layers:
                
            if l not in ebm_params[esm][f'{k}layer'].keys():
                continue      
            elif len(ebm_params[esm][f'{k}layer'][l].keys()) == 0:
                continue

            rmse_err_full = np.sqrt(np.mean((fair_4xco2_output[esm][f'{k}layer'][l
                        ] - fair_4xco2_output[esm]['ESM'])**2))
        
            rmse_err_last100 = np.sqrt(np.mean((fair_4xco2_output[esm][f'{k}layer'][l
                        ][:-100] - fair_4xco2_output[esm]['ESM'][:-100])**2))
            
                        
            ax.scatter(l/l0, rmse_err_full.data, color = layer_attrs[f'{k}layer'][0][0], marker=layer_attrs[f'{k}layer'][0][1])
            ax2.scatter(l/l0, rmse_err_last100.data, color = layer_attrs[f'{k}layer'][1][0], marker=layer_attrs[f'{k}layer'][1][1])
        
        
            if l in ebm_params[esm]['3layer'].keys() and l in ebm_params[esm]['2layer'].keys():
                if len(ebm_params[esm]['3layer'][l]) != 0 and len(ebm_params[esm]['2layer'][l]) != 0:
    
                    rmse_errs[f'{k}layer']['Full'].append(rmse_err_full.data)
                    rmse_errs[f'{k}layer']['Last100'].append(rmse_err_last100.data)

        if l in ebm_params[esm]['3layer'].keys() and l in ebm_params[esm]['2layer'].keys():
            
            data_2layer = rmse_errs['2layer']['Full'][-1]
            data_3layer = rmse_errs['3layer']['Full'][-1]

            linestyle = '--'
            if data_2layer < data_3layer:
                linestyle = 'solid'
                
            ax.vlines(x = l/l0, ymin=data_2layer, ymax=data_3layer, linestyle = linestyle, color='grey')
            
        
        
ax.set_ylabel('RMSE (K)')  
# ax.set_ylabel('Overall MAE (K)')  
# ax2.set_ylabel('Last 100yr MAE (K)') 
ax.set_title('Root-Mean-Square Error')  

# ax.yaxis.label.set_color('red')
# ax2.yaxis.label.set_color('blue')

# ax.spines['left'].set_color('red')
# ax2.spines['right'].set_color('blue')

ax.set_xlabel('Training / Total')  

ax_ylim = ax.get_ylim()
ax2_ylim = ax2.get_ylim()

ylim = [np.amin((ax_ylim, ax2_ylim)), np.amax((ax_ylim, ax2_ylim))]

ax.set_ylim(ylim)
ax2.set_ylim(ylim)
ax2.set_yticks([])


handles = []

for k in layers:
        
    handles.append(Line2D([0], [0], color = layer_attrs[f'{k}layer'][0][0], marker=layer_attrs[f'{k}layer'][0][1
                   ], label=f'{k} layer Overall', linestyle=''))
    
    handles.append(Line2D([0], [0], color = layer_attrs[f'{k}layer'][1][0], marker=layer_attrs[f'{k}layer'][1][1
                   ], label=f'{k} layer last 100', linestyle=''))

handles.append(Line2D([0], [0], color ='grey', label='3layer > 2layer', linestyle='solid'))
handles.append(Line2D([0], [0], color ='grey', label='2layer > 3layer', linestyle='--'))


ax.legend(handles=handles, ncol=3, loc = 'upper right')

plt.tight_layout()
plt.savefig(
    "../figures/for_paper/supplement/S3_4xCO2_RMSE.png"
)


#%%

for n in ['Full', 'Last100']:
        
    err_2_minus_3 = np.asarray(rmse_errs['2layer'][n]) - np.asarray(rmse_errs['3layer'][n])
    print(f'Error in 2 cf 3 layers {n}: {np.around(np.mean(err_2_minus_3), decimals=2)}')
    
    perc_better_with_3 = 100*(np.sum(np.array(err_2_minus_3) >= 0)/err_2_minus_3.shape[0])
    print(f'3 layer better for {n} in {np.around(np.mean(perc_better_with_3), decimals=0) } of cases')
            