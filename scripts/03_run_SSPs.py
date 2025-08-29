"""
This script runs the SSPs with the calibrated EBM parameters and makes
plots 2 and 3 in the paper

Author: Chris Wells - August 2025
"""
from fair import FAIR
from fair.interface import fill, initialise
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib as mpl

layers = [2, 3]

with open('../data/output/ebm_params.pkl', 'rb') as handle:
    ebm_params = pickle.load(handle)

esms = list(ebm_params.keys())
esms.sort()

scenarios = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp534-over', 'ssp585']

# forcing csvs from https://zenodo.org/records/5705391
scen_tables = {
    "ssp119":"A3.4a",
    "ssp126":"A3.4b",
    "ssp245":"A3.4c",
    "ssp370":"A3.4d",
    "ssp434":"A3.4x",
    "ssp460":"A3.4x",
    "ssp534-over":"A3.4x",
    "ssp585":"A3.4e",
    }

variables = {
    'GMST':'K', 
    'SLR':'m', 
    }

y_start = 1750
y_end = 2500

j_to_metres_slr = 0.0975*1E-24

fair_ssp_output = {}

with open('../data/misc/colors_pd.pkl', 'rb') as handle:
    colors_pd = pickle.load(handle)

scen_names = {
"ssp119":"AR6-SSP1-1.9",
"ssp126":"AR6-SSP1-2.6",
"ssp245":"AR6-SSP2-4.5",
"ssp370":"AR6-SSP3-7.0",
"ssp434":"AR6-SSP4-3.4",
"ssp460":"AR6-SSP4-6.0",
"ssp534-over":"AR6-SSP5-3.4-OS",
"ssp585":"AR6-SSP5-8.5",
    }

#%%
for esm in esms:
    print(esm)
    fair_ssp_output[esm] = {}
    for k in layers:
    
        fair_ssp_output[esm][f'{k}layer'] = {}
        fair_ssp_output[esm][f'{k}layer']['GMST'] = {}
        fair_ssp_output[esm][f'{k}layer']['SLR'] = {}
    
        for l in ebm_params[esm][f'{k}layer'].keys():
            if len(ebm_params[esm][f'{k}layer'][l].keys()) == 0:
                continue
            
            f = FAIR(n_layers=k)
            f.define_time(y_start, y_end, 1)
            f.timebounds
            
            f.define_scenarios(scenarios)     
            
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
            
            for scen in scenarios:
                df_forc = pd.read_csv(
                    f'../data/ssp_forcings/table_{scen_tables[scen]}_{scen}_ERF_1750-2500_best_estimate.csv')
                    
                fill(f.forcing, df_forc['total'].values, config=configs[0], 
                     specie='CO2', scenario = scen)
            
            f.fill_species_configs()
            
            fill(f.species_configs['tropospheric_adjustment'], 0, specie='CO2')
            
            f.run()
            
            fair_ssp_output[esm][f'{k}layer']['GMST'][l] = f.temperature.loc[dict(layer=0)][:,:,0]
            fair_ssp_output[esm][f'{k}layer']['SLR'][l] = f.ocean_heat_content_change[:,:,0]*j_to_metres_slr

#%%

pd_warming = 1.24 # IGCC 2015-24

pd_idxs = [2015 - y_start, 2024 - y_start + 1]

l_roll = 20

GWLs = [1.5, 2, 2.5, 3, 4]

peak_scens = ['ssp119', 'ssp534-over']

crossing_times = {}
peak_warming = {}
for esm in fair_ssp_output.keys():
    crossing_times[esm] = {}
    peak_warming[esm] = {}
    for k in layers:
        crossing_times[esm][f'{k}layer'] = {}
        peak_warming[esm][f'{k}layer'] = {}
        for l in fair_ssp_output[esm][f'{k}layer']['GMST'].keys():
            crossing_times[esm][f'{k}layer'][l] = {}
            peak_warming[esm][f'{k}layer'][l] = {}
            for s_i, scen in enumerate(scenarios):
                crossing_times[esm][f'{k}layer'][l][scen] = {}
                
                delT = fair_ssp_output[esm][f'{k}layer']['GMST'][l][:,s_i]
                
                delT_offset = delT - np.mean(delT[pd_idxs[0]:pd_idxs[1]]) + pd_warming
                
                delT_offset_rolling = np.convolve(delT_offset, np.ones(l_roll)/l_roll, mode='valid')
                
                if scen in peak_scens:
                    Tpeak = np.amax(delT_offset_rolling)
                    
                    np.where(delT_offset_rolling == Tpeak)
                    
                    peak_warming[esm][f'{k}layer'][l][scen] = Tpeak
                    
                    if np.where(delT_offset_rolling == Tpeak)[0][0] + 1 == delT_offset_rolling.shape:
                        print(f'{esm} {l}')

                for gwl in GWLs:
                    if np.amax(delT_offset_rolling) < gwl:
                        continue
                    
                    idx_cross = np.argmax(delT_offset_rolling>gwl)
                    crossing_times[esm][f'{k}layer'][l][scen][gwl] = idx_cross + y_start
                   
#%%
#Figure 2

comps = [
    [[150, 3], [150, 2]],# [[l1, k1], [l2, k2]]
    [[300, 3], [150, 3]],
    [[900, 3], [150, 3]],
    ]

linestyles = ['solid', 'dashed', 'dotted']

idx_end = np.where(f.timebounds == 2500)[0][0]

time_roll = np.convolve(f.timebounds[:idx_end], np.ones(l_roll)/l_roll, mode='valid')

fig, axs = plt.subplots(5, 4, figsize=(12, 15))
axs = axs.ravel()

for i, esm in enumerate(esms):
    for c_i, comp in enumerate(comps):

        l1 = comp[0][0]
        k1 = comp[0][1]
        
        l2 = comp[1][0]
        k2 = comp[1][1]
        
        if l1 in fair_ssp_output[esm][f'{k1}layer']['GMST'
            ].keys() and l2 in fair_ssp_output[esm][f'{k2}layer']['GMST'].keys():
        
            for s_i, scen in enumerate(scenarios):
                
                gmst_roll = np.convolve(fair_ssp_output[esm][f'{k1}layer']['GMST'][l1][:idx_end,s_i] - fair_ssp_output[
                    esm][f'{k2}layer']['GMST'][l2][:idx_end,s_i], np.ones(l_roll)/l_roll, mode='valid')

                axs[i].plot(time_roll, gmst_roll, color=colors_pd.loc[
                        colors_pd['name'] == scen_names[scen]]['color'].values[0],
                        linestyle = linestyles[c_i])
                
    axs[i].set_title(f'{esm}')
    axs[i].set_xlabel('Years')
    if i % 4 == 0:
        axs[i].set_ylabel('K')
    axs[i].axhline(y=0, color='grey', linestyle='--')
    
handles = []
for scen in scenarios:
    handles.append(Line2D([0], [0], label=f'{scen}', color=colors_pd.loc[
        colors_pd['name'] == scen_names[scen]]['color'].values[0]))
for c_i, comp in enumerate(comps):
    l1 = comp[0][0]
    k1 = comp[0][1]
    
    l2 = comp[1][0]
    k2 = comp[1][1]
    
    handles.append(Line2D([0], [0], label=f'{l1} (k={k1}) - {l2} (k={k2})', linestyle = linestyles[c_i], color='grey'))

fig.legend(handles=handles, ncol=2, bbox_to_anchor=[0.6, 0.11], loc='center', prop={'size': 12})

for j in np.arange(len(ebm_params.keys()), 20):
    axs[j].set_visible(False)

plt.tight_layout()
plt.savefig(
    "../figures/for_paper/fig2_timeseries.png"
)

#%%

#Figure S2

combo_list = [c for cs in comps for c in cs]
combo_set = []
for combo in combo_list:
    if combo not in combo_set:
        combo_set.append(combo)

linestyles = ['solid', 'dashed', 'dotted', 'dashdot']

idx_end = np.where(f.timebounds == 2500)[0][0]

fig, axs = plt.subplots(5, 4, figsize=(12, 15))
axs = axs.ravel()

for i, esm in enumerate(ebm_params.keys()):
    for c_i, combo in enumerate(combo_set):

        l = combo[0]
        k = combo[1]
        if l in fair_ssp_output[esm][f'{k}layer']['GMST'].keys():
        
            for s_i, scen in enumerate(scenarios):
                
                gmst_roll = np.convolve(fair_ssp_output[esm][f'{k}layer']['GMST'][l][:idx_end,s_i
                         ], np.ones(l_roll)/l_roll, mode='valid')

                axs[i].plot(time_roll, gmst_roll, color=colors_pd.loc[
                        colors_pd['name'] == scen_names[scen]]['color'].values[0],
                        linestyle = linestyles[c_i])
                
    axs[i].set_title(f'{esm}')
    axs[i].set_xlabel('Years')
    if i % 4 == 0:
        axs[i].set_ylabel('K')
    axs[i].axhline(y=0, color='grey', linestyle='--')
    
handles = []
for scen in scenarios:
    handles.append(Line2D([0], [0], label=f'{scen}', color=colors_pd.loc[
        colors_pd['name'] == scen_names[scen]]['color'].values[0]))
for c_i, combo in enumerate(combo_set):
    l = combo[0]
    k = combo[1]
    handles.append(Line2D([0], [0], label=f'{l} (k={k})', linestyle = linestyles[c_i], color='grey'))

fig.legend(handles=handles, ncol=2, bbox_to_anchor=[0.6, 0.11], loc='center', prop={'size': 12})

for j in np.arange(len(ebm_params.keys()), 20):
    axs[j].set_visible(False)

plt.tight_layout()
plt.savefig(
    "../figures/for_paper/supplement/S2_timeseries.png"
)


#%%

# Figure 3

years = [2050, 2100, 2300, 2500]
# comps = [
#     [[300, 3], [150, 3]],# [[l1, k1], [l2, k2]]
#     [[900, 3], [150, 3]],
#     [[900, 4], [150, 3]],
#     ]

comps = [
    [[150, 3], [150, 2]],# [[l1, k1], [l2, k2]]
    [[300, 3], [150, 3]],
    [[900, 3], [150, 3]],
    ]


scens_to_plot = ['ssp119', 'ssp245', 'ssp534-over', 'ssp585']

# want only the ESMs with the all lengths
esms = []
ls = []
for c in comps:
    ls.append(c[0][0])
    ls.append(c[1][0])

lmax = np.amax(ls)
for esm in fair_ssp_output.keys():
    if lmax in fair_ssp_output[esm]['3layer']['GMST'
          ].keys() and lmax in fair_ssp_output[esm]['2layer']['GMST'].keys():
        esms.append(esm)

hatches = ['', '///', 'xx']

ylims = {
    'GMST':[-1, 2], 
    'SLR':[-0.1, 0.6], 
    }

fig, axs = plt.subplots(2, len(years), figsize=(4*len(years), 8))

for v_i, var in enumerate(variables.keys()):
    for y_i, y in enumerate(years):
        y_idx = y - y_start

        x = 0
        s_count = 0
        for s_i, scen in enumerate(scenarios):
            if scen in scens_to_plot:
                s_count += 1
                x += 0.7
                
                for c_i, comp in enumerate(comps):
                    x += 1

                    l1 = comp[0][0]
                    k1 = comp[0][1]
                    
                    l2 = comp[1][0]
                    k2 = comp[1][1]
                    
                    plot_data = np.full(len(esms), np.nan)
                    for e_i, esm in enumerate(esms):
                        
                        expt_cf2020 = fair_ssp_output[esm][f'{k1}layer'][var][l1
                                    ][y_idx][s_i] - fair_ssp_output[esm][f'{k1}layer'][var][l1
                                    ][2020 - y_start][s_i]
                        
                        ctrl_cf2020 = fair_ssp_output[esm][f'{k2}layer'
                         ][var][l2][y_idx][s_i] - fair_ssp_output[esm][f'{k2}layer'
                          ][var][l2][2020 - y_start][s_i]
                                                                       
                        plot_data[e_i] = expt_cf2020 - ctrl_cf2020                                   
                                                                                            
                    c = colors_pd.loc[colors_pd['name'
                    ] == scen_names[scen]]['color'].values[0]  
                    
                    axs[v_i, y_i].boxplot(plot_data, positions = [x], widths = [0.5],
                                     whis = [0, 100], showfliers=False,
                                          patch_artist=True,
                    boxprops=dict(hatch = hatches[c_i], facecolor=c, alpha=0.5),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color='white'),
                                          )
                    
                    # print median, range for specific cases (for abstract)
                    if y == 2100 and scen == 'ssp585' and var == 'GMST' and c_i == 2:
                        print(f'Median: {np.median(plot_data)} for {y}')
                        print(f'25, 75 %ile: {np.percentile(plot_data, [25, 75])} for {y}')
        
                    if y == 2500 and scen == 'ssp585' and var == 'SLR' and c_i == 2:
                        print(f'Median: {np.median(plot_data)} for {y}')
                        print(f'25, 75 %ile: {np.percentile(plot_data, [25, 75])} for {y}')
        
        
        xticks = axs[v_i, y_i].get_xticks()
        axs[v_i, y_i].set_xticks(xticks[::len(comps)] + 0.5, labels=scens_to_plot, rotation=45)
        
        if y_i >0:
            axs[v_i, y_i].set_yticklabels([])
        else:
            axs[v_i, y_i].set_ylabel(f'{variables[var]} cf 2020')
        
        axs[v_i, y_i].axhline(y=0, linestyle='--', color='grey')
        axs[v_i, y_i].set_title(f'{y}')
        
        axs[v_i, y_i].set_ylim(ylims[var])
            
        handles = []
        for s_i, scen in enumerate(scenarios):
            if scen in scens_to_plot:
                c = colors_pd.loc[colors_pd['name'
                ] == scen_names[scen]]['color'].values[0] 
                
                handles.append(mpatches.Patch(color=c, alpha=0.5, label=f'{scen}'))
        

        for c_i, comp in enumerate(comps):
            l1 = comp[0][0]
            k1 = comp[0][1]
            
            l2 = comp[1][0]
            k2 = comp[1][1]  

            c = 'grey'            

            patch = mpatches.Patch(hatch = hatches[c_i], facecolor=c, alpha=0.5,
                          label=f'{l1} (k={k1}) - {l2} (k={k2})')
            
            patch.set_edgecolor('black')
            
            handles.append(patch)
        axs[1,0].legend(handles=handles)
            
plt.subplots_adjust(wspace=0, hspace=0.4)

plt.suptitle(f'GMST, SLR cf 2020 ({len(esms)} ESMs)')
        
plt.tight_layout()
plt.savefig(
    "../figures/for_paper/fig3_boxes.png"
)
 

#%%
# Figure 4

cmap = mpl.colormaps['inferno']

colors = {
    "1.5":cmap(0.15),
    "2":cmap(0.35),
    "2.5":cmap(0.5),
    "3":cmap(0.7),
    "4":cmap(0.9),
    }

comps = [
    [[150, 3], [150, 2]],# [[l1, k1], [l2, k2]]
    [[300, 3], [150, 3]],
    [[900, 3], [150, 3]],
    ]

markers = ['o', 'x', 's']
hatches = ['', '///', 'xx']

s_scaling = 0.3

esms = []
ls = []
for c in comps:
    ls.append(c[0][0])
    ls.append(c[1][0])

lmax = np.amax(ls)
for esm in fair_ssp_output.keys():
    if lmax in fair_ssp_output[esm]['3layer']['GMST'
          ].keys() and lmax in fair_ssp_output[esm]['3layer']['GMST'].keys():
        esms.append(esm)

handles = []
for size in [5, 25, 50]:
    handles.append(Line2D([0], [0], label=f'{size}', color='grey', marker='^', markersize=s_scaling*size, linestyle=''))
    
for gwl in GWLs:
    handles.append(mpatches.Patch(facecolor=colors[f'{gwl}'], 
                  label=f'{gwl}K'))
    
handles.append(Line2D([], [], label='Median', color='black'))

for c_i, comp in enumerate(comps):
    l1 = comp[0][0]
    k1 = comp[0][1]
    
    l2 = comp[1][0]
    k2 = comp[1][1]  
    
    handles.append(Line2D([0], [0], label=f'{l1} (k={k1}) - {l2} (k={k2})', color='grey', marker=markers[c_i], linestyle=''))

fig, axs = plt.subplots(1, 2, figsize=(12, 6), width_ratios=[5,2])

axs[0].set_title(f'Crossing time differences ({len(esms)} ESMs)')
axs[0].set_xlabel('GWL (K)')
axs[0].set_ylabel('Years')
axs[0].axhline(y=0, color='grey', linestyle='--')

yticks = np.arange(-15, 12, 3)
axs[0].set_yticks(yticks, labels=yticks)

xticks_list = []
x = 0

for g_i, gwl in enumerate(GWLs):
    x += 1
    xticks_list.append(x)
    for c_i, comp in enumerate(comps):
        x += 1

        l1 = comp[0][0]
        k1 = comp[0][1]
        
        l2 = comp[1][0]
        k2 = comp[1][1]
        
        plot_data = []
        for esm in esms:
            for s_i, scen in enumerate(crossing_times[esm][f'{k1}layer'][l1].keys()):
                if gwl in crossing_times[esm][f'{k1}layer'][l1][scen
                  ].keys() and gwl in crossing_times[esm][f'{k2}layer'][l2][scen].keys():
                    
                    c = colors_pd.loc[colors_pd['name'
                    ] == scen_names[scen]]['color'].values[0] 
                    
                    crossing_dif = crossing_times[esm][f'{k1}layer'][l1][scen][gwl
                             ] - crossing_times[esm][f'{k2}layer'][l2][scen][gwl]
                    
                    plot_data.append(crossing_dif)
                    
        c = colors[f'{gwl}']
        
        for s in set(plot_data):
            n_s = np.sum(plot_data == s)
        
            axs[0].plot(x, s, markersize=s_scaling*n_s, color=c, marker=markers[c_i])                      
                  
        axs[0].plot([x-0.4, x+0.4], [np.median(plot_data), np.median(plot_data)], color='black')            
        axs[0].text(x, -14, f'{len(plot_data)}', horizontalalignment='center',
                bbox=dict(facecolor='white', edgecolor='white', alpha=1)
                )
        
 
xticks = axs[0].get_xticks()
xticks_new = np.asarray(xticks_list) + 0.5*(len(comps)-1) + 1
axs[0].set_xticks(xticks_new, labels=GWLs)

axs[0].set_ylim([-15, 10])  
axs[0].set_xlim([xticks_new[0]-2, xticks_new[-1]+len(comps) - 1])  

axs[0].legend(handles=handles, ncol=4)
 
x = 0
for s_i, scen in enumerate(peak_scens):
    x += 1
    for c_i, comp in enumerate(comps):
        x += 1

        l1 = comp[0][0]
        k1 = comp[0][1]
        
        l2 = comp[1][0]
        k2 = comp[1][1]
        
        plot_data = []
        for esm in esms:
            if scen in peak_warming[esm][f'{k1}layer'][l1
                   ].keys() and scen in peak_warming[esm][f'{k1}layer'][l2].keys():
                
                c = colors_pd.loc[colors_pd['name'
                ] == scen_names[scen]]['color'].values[0] 
                
                peak_dif = peak_warming[esm][f'{k1}layer'][l1][scen
                               ] - peak_warming[esm][f'{k2}layer'][l2][scen]
                
                plot_data.append(peak_dif)
                
        c = colors_pd.loc[colors_pd['name'
        ] == scen_names[scen]]['color'].values[0]
        
        axs[1].boxplot(np.asarray(plot_data), positions = [x], widths = [0.5],
                         whis = [0, 100], showfliers=False,
                              patch_artist=True,
        boxprops=dict(hatch = hatches[c_i], facecolor=c, alpha=0.5),
        capprops=dict(color=c),
        whiskerprops=dict(color=c),
        flierprops=dict(color=c, markeredgecolor=c),
        medianprops=dict(color='white'),
                              )
        

        # print median, range for specific cases (for main text)
        if c_i == 2:
            print(f'Median {scen}: {np.median(plot_data)}')
            print(f'25, 75 %ile {scen}: {np.percentile(plot_data, [25, 75])}')

 
axs[1].set_title(f'Peak GMST differences ({len(esms)} ESMs)')
axs[1].set_ylabel('K')
axs[1].set_xticks(np.arange(len(peak_scens))*(1 + len(comps)) + 0.5*len(comps) + 1.5, labels=peak_scens)

axs[1].axhline(y=0, color='grey', linestyle='--')
axs[1].set_ylim([-0.2, 0.5])  


handles = []
for s_i, scen in enumerate(scenarios):
    if scen in peak_scens:
        c = colors_pd.loc[colors_pd['name'
        ] == scen_names[scen]]['color'].values[0] 
        
        handles.append(mpatches.Patch(color=c, label=f'{scen}', alpha=0.5))


for c_i, comp in enumerate(comps):
    l1 = comp[0][0]
    k1 = comp[0][1]
    
    l2 = comp[1][0]
    k2 = comp[1][1]  
    
    patch = mpatches.Patch(hatch = hatches[c_i], facecolor='grey', 
                  label=f'{l1} (k={k1}) - {l2} (k={k2})')
    
    patch.set_edgecolor('black')
    
    handles.append(patch)

axs[1].legend(handles=handles)

axs[0].text(-0.05, 1.02, 'a)', transform=axs[0].transAxes)
axs[1].text(-0.15, 1.02, 'b)', transform=axs[1].transAxes)

plt.tight_layout()
plt.savefig(
    "../figures/for_paper/fig4_crossing_times.png"
)

