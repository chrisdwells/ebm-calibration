"""
This script finds the ECS and TCR values associated with the calibrated EBM 
parameters and makes plots 1 and S4 in the paper

Author: Chris Wells - August 2025
"""
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D
from fair import FAIR
from fair.interface import fill, initialise
import scipy.linalg
import scipy.sparse.linalg
import scipy.stats
import pandas as pd

# Estimates ECS, TCR, other EBM properties; plots 1, S4

layers = [2, 3]

with open('../data/output/ebm_params.pkl', 'rb') as handle:
    ebm_params = pickle.load(handle)

markers_list = list(Line2D.markers.keys())
markers_list.remove(',')

ebm_outputs = {}

for k in layers:
    ebm_outputs[f'{k}layer'] = {}
    for esm in ebm_params.keys():
        ebm_outputs[f'{k}layer'][esm] = {}
        for l in ebm_params[esm][f'{k}layer'].keys():
            if len(ebm_params[esm][f'{k}layer'][l].keys()) == 0:
                continue
            
            ebm_outputs[f'{k}layer'][esm][l] = {}
            
            f = FAIR(n_layers=k)
            f.define_time(0, 1, 1)
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

            fill(f.climate_configs['forcing_4co2'], ebm_params[esm][f'{k}layer'][l]["F4xCO2"], config=configs[0])

            f.run()
            
            parameters = ebm_params[esm][f'{k}layer'][l]
            
            gamma = 1000
            C = ebm_params[esm][f'{k}layer'][l]["C"]
            kappa = ebm_params[esm][f'{k}layer'][l]["kappa"]
            epsilon = ebm_params[esm][f'{k}layer'][l]["epsilon"]
            F_4xCO2 = ebm_params[esm][f'{k}layer'][l]["F4xCO2"]
            
            if k == 2:
                A = np.array([
                     [-gamma,                                   0,                     0],
                     [1/C[0], -(kappa[0] + epsilon*kappa[1])/C[0], epsilon*kappa[1]/C[0]],
                     [     0,                       kappa[1]/C[1],        -kappa[1]/C[1]]
                ])
            elif k == 3:
                A = np.array([
                     [-gamma,                           0,                                   0,                     0],
                     [1/C[0], -(kappa[0] + kappa[1])/C[0],                       kappa[1]/C[0],                     0],
                     [0,                    kappa[1]/C[1], -(kappa[1] + epsilon*kappa[2])/C[1], epsilon*kappa[2]/C[1]],
                     [0,                                0,                       kappa[2]/C[2],        -kappa[2]/C[2]]
                ])
            else:
                raise ValueError("Number of boxes must be 2 or 3.")
    
    
            eb_matrix = A
            
            eb_matrix_eigenvalues, eb_matrix_eigenvectors = scipy.linalg.eig(
                eb_matrix[1:, 1:]
            )
            timescales = -1 / (np.real(eb_matrix_eigenvalues))
            response_coefficients = (
                timescales
                * (
                    eb_matrix_eigenvectors[0, :]
                    * scipy.linalg.inv(eb_matrix_eigenvectors)[:, 0]
                )
                / C[0]
            )
        
            ebm_outputs[f'{k}layer'][esm][l]['ECS'] = f.ebms.ecs
            ebm_outputs[f'{k}layer'][esm][l]['TCR'] = f.ebms.tcr

            ebm_outputs[f'{k}layer'][esm][l]['timescales'] = timescales
            ebm_outputs[f'{k}layer'][esm][l]['response_coefficients'] = response_coefficients


#%%
# Figure 1

cmap = mpl.colormaps['plasma']

k = 3

fig = plt.figure(figsize=(6, 6))

fig.add_subplot(111)
    
for e_i, esm in enumerate(ebm_params.keys()):
    plot_data_ls = list(ebm_params[esm][f'{k}layer'].keys())
    plot_data = []
    
    for l in ebm_params[esm][f'{k}layer'].keys():
        
        if esm == 'CNRMCM61' and l == 1850:
            plot_data_ls.remove(l)

            continue
        elif len(ebm_params[esm][f'{k}layer'][l].keys()) == 0:
            plot_data_ls.remove(l)
        else:
            plot_data.append(ebm_outputs[f'{k}layer'][esm][l]['ECS'])
    
    idxs = np.argsort(plot_data_ls)
    
    plt.scatter(np.asarray(plot_data_ls)[idxs], np.asarray(plot_data)[idxs], 
         color=cmap(e_i/len(ebm_params.keys())), marker=markers_list[e_i],
             s=100)
    
    plt.plot(np.asarray(plot_data_ls)[idxs], np.asarray(plot_data)[idxs], color=cmap(e_i/len(ebm_params.keys())))

plt.ylabel('ECS (K)')

plt.xlabel('Training length (years)')

handles = []
for e_i, esm in enumerate(ebm_params.keys()):
    handles.append(Line2D([], [], label=f'{esm}', color=cmap(e_i/len(ebm_params.keys())), marker=markers_list[e_i]))
plt.legend(handles=handles, ncol=2)

plt.suptitle(f'ECS against Training length for 3-layer fit ({len(list(ebm_params.keys()))} ESMs)')

plt.xticks(
    ticks=[0, 150, 300, 500, 900, 1000, 2000, 3000, 4000, 5000, 6000], 
    labels=['0', '', '', '', '', '1000', '2000', '3000', '4000', '5000', '6000'])


plt.tight_layout()
plt.savefig(
    "../figures/for_paper/fig1_ECS.png"
)

#%%
# Fig S4 timescales, response
cmap = mpl.colormaps['plasma']

fig, axs = plt.subplots(1, 3, figsize=(14, 9))


for e_i, esm in enumerate(ebm_outputs['3layer'].keys()):
    plot_data_ls = list(ebm_params[esm]['3layer'].keys())
    plot_data = {}
    
    plot_data['3layer'] = {}
    plot_data['3layer']['Magnitude'] = []
    plot_data['3layer']['Fraction'] = []

    for l in plot_data_ls:
        if len(ebm_params[esm]['3layer'][l].keys()) == 0:
            plot_data['3layer']['Magnitude'].append(np.nan)
            plot_data['3layer']['Fraction'].append(np.nan)
            
        else:
            plot_data['3layer']['Magnitude'].append(ebm_outputs[
                '3layer'][esm][l]['timescales'][-1])
            plot_data['3layer']['Fraction'].append(ebm_outputs[
                '3layer'][esm][l]['timescales'][-1]/l)

    idxs = np.argsort(plot_data_ls)
    
    axs[0].scatter(np.asarray(plot_data_ls)[idxs], np.asarray(plot_data['3layer']['Magnitude'])[idxs], 
         color=cmap(e_i/len(ebm_params.keys())), marker=markers_list[e_i],
             s=100)
    
    axs[0].plot(np.asarray(plot_data_ls)[idxs], np.asarray(plot_data['3layer']['Magnitude'])[idxs], color=cmap(e_i/len(ebm_params.keys())))

    axs[1].scatter(np.asarray(plot_data_ls)[idxs], np.asarray(plot_data['3layer']['Fraction'])[idxs], 
         color=cmap(e_i/len(ebm_params.keys())), marker=markers_list[e_i],
             s=100)
    
    axs[1].plot(np.asarray(plot_data_ls)[idxs], np.asarray(plot_data['3layer']['Fraction'])[idxs], color=cmap(e_i/len(ebm_params.keys())))


axs[0].set_title('Deep ocean timescale 3 layers')
axs[1].set_title('Deep ocean timescale on L 3 layers')


axs[0].set_ylabel('Years')
axs[0].set_xlabel('Training length (years)')
axs[1].set_xlabel('Training length (years)')

axs[0].text(-0.08, 1.01, 'a)', transform=axs[0].transAxes)
axs[1].text(-0.08, 1.01, 'b)', transform=axs[1].transAxes)
axs[2].text(-0.08, 1.01, 'c)', transform=axs[2].transAxes)

handles = []
for e_i, esm in enumerate(ebm_params.keys()):
    handles.append(Line2D([], [], label=f'{esm}', color=cmap(e_i/len(ebm_params.keys())), marker=markers_list[e_i]))
axs[1].legend(handles=handles, ncol=2)


for e_i, esm in enumerate(ebm_outputs['3layer'].keys()):
    for l in ebm_outputs['3layer'][esm].keys():
        
        tau_sum = np.sum(ebm_outputs['3layer'][esm][l]['response_coefficients'])
        for t_i in np.arange(3):
            
            tau_frac = ebm_outputs['3layer'][esm][l][
                'response_coefficients'][t_i]/tau_sum
            
            axs[2].plot(l, tau_frac, marker = markers_list[t_i], 
                          color=f'C{t_i}')

axs[2].set_title('Normalised response coefficients 3 layer')
axs[2].set_xlabel('Training length (years)')

handles = []
for t_i in np.arange(3):
    handles.append(Line2D([0], [0], label=f'Layer {t_i+1}', marker = markers_list[t_i], 
                  color=f'C{t_i}', linestyle=''))
axs[2].legend(handles=handles)

plt.tight_layout()
plt.savefig(
    "../figures/for_paper/supplement/S4_timescales.png"
)

#%%

# Find how many ESMs see ECS increase when going from L=150 years to 900, lmax

dif_3_900_vs_3_150 = []
dif_3_max_vs_3_150 = []

for esm in ebm_outputs['3layer'].keys():
    
    if 900 in ebm_outputs['3layer'][esm].keys():
        
        dif = ebm_outputs['3layer'][esm][900]['ECS'].values[0] - ebm_outputs['3layer'][esm][150]['ECS'].values[0]
        if dif < 0:
            print(f'{esm} 900')
        
        dif_3_900_vs_3_150.append(dif)
    
    lmax = np.amax(np.asarray(list(ebm_outputs['3layer'][esm].keys())))

    dif = ebm_outputs['3layer'][esm][lmax]['ECS'].values[0] - ebm_outputs['3layer'][esm][150]['ECS'].values[0]
    if dif < 0:
        print(f'{esm} {lmax}')
    
    dif_3_max_vs_3_150.append(dif)
    

print(f'{(np.asarray(dif_3_900_vs_3_150)>0).sum()} out of {len(dif_3_900_vs_3_150)} 900 vs 150')
print(f'{(np.asarray(dif_3_max_vs_3_150)>0).sum()} out of {len(dif_3_max_vs_3_150)} lmax vs 150')

#%%

# create the CSV for Table S1 and the output info
# Table S1: ESM, Length, Data source
# Output: ESM, k, L, ECS, TCR, F4xCO2, thermal parameters (C, kappa, efficacy), 
# IRF parameters (d, A)

# Table S1

source_not_longrunmip = {
        'MIROC32':"longRunMIP (Estimated; see text)",
        'CESM2':"CMIP6 (10.22033/ESGF/CMIP6.7519)",
        'GFDL-CM2.1p1':"He et al., (2025)",
        'GFDL-ESM4':"Dunne et al., (2020)",
        'IPSL-CM6A-LR':"CMIP6 (10.22033/ESGF/CMIP6.5109)",
        'NorESM2-LM':"CMIP6 (10.22033/ESGF/CMIP6.7836)",
                    }


df_s1 = pd.DataFrame(columns = ['ESM', 'Length (years)', 'Data Source'])
df_output = pd.DataFrame(columns = ['ESM', 'Layers', 'Calibration length',
                                    'ECS', 'TCR', 'F4xCO2', 'Thermal C1', 'Thermal C2', 'Thermal C3', 
                                    'Thermal kappa1', 'Thermal kappa2', 'Thermal kappa3', 'Thermal Efficacy',
                                    'IRF tau1', 'IRF tau2', 'IRF tau3', 'IRF q1', 'IRF q2', 'IRF q3'
                                    ])

for esm in ebm_outputs['3layer'].keys():
    lmax = np.amax(np.asarray(list(ebm_outputs['3layer'][esm].keys())))
    
    source = 'longRunMIP'
    if esm in source_not_longrunmip.keys():
        source = source_not_longrunmip[esm]
    row_s1 = [esm, lmax, source]
    
    df_s1.loc[len(df_s1)] = row_s1
        
    for k in layers:
        
        ls = np.asarray(list(ebm_outputs['3layer'][esm].keys()))
        
        ls.sort()
        
        for l in ls:
            ecs = np.around(ebm_outputs[f'{k}layer'][esm][l]['ECS'].values[0], decimals=4)
            tcr = np.around(ebm_outputs[f'{k}layer'][esm][l]['TCR'].values[0], decimals=4)
            
            f4xco2 = np.around(ebm_params[esm][f'{k}layer'][l]["F4xCO2"], decimals=4)

            c1 = np.around(ebm_params[esm][f'{k}layer'][l]["C"][0], decimals=4)
            c2 = np.around(ebm_params[esm][f'{k}layer'][l]["C"][1], decimals=4)

            k1 = np.around(ebm_params[esm][f'{k}layer'][l]["kappa"][0], decimals=4)
            k2 = np.around(ebm_params[esm][f'{k}layer'][l]["kappa"][1], decimals=4)

            efficacy = np.around(ebm_params[esm][f'{k}layer'][l]["epsilon"], decimals=4)
            
            tau1 = np.around(ebm_outputs[f'{k}layer'][esm][l]['timescales'][0], decimals=4)
            tau2 = np.around(ebm_outputs[f'{k}layer'][esm][l]['timescales'][1], decimals=4)

            q1 = np.around(ebm_outputs[f'{k}layer'][esm][l]['response_coefficients'][0], decimals=4)
            q2 = np.around(ebm_outputs[f'{k}layer'][esm][l]['response_coefficients'][1], decimals=4)

            if k == 3:
                c3 = np.around(ebm_params[esm][f'{k}layer'][l]["C"][2], decimals=4)
                k3 = np.around(ebm_params[esm][f'{k}layer'][l]["kappa"][2], decimals=4)
                tau3 = np.around(ebm_outputs[f'{k}layer'][esm][l]['timescales'][2], decimals=4)
                q3 = np.around(ebm_outputs[f'{k}layer'][esm][l]['response_coefficients'][2], decimals=4)
            else:
                c3, k3, tau3, q3 = ['N/A', 'N/A', 'N/A', 'N/A']

            row_output = [esm, k, l, ecs, tcr, f4xco2, c1, c2, c3, k1, k2, k3,
                          efficacy, tau1, tau2, tau3, q1, q2, q3]
            
            df_output.loc[len(df_output)] = row_output

df_s1.to_csv('../data/output/table_s1.csv', index=False)
df_output.to_csv('../data/output/table_extended1.csv', index=False)

        

