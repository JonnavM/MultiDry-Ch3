#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 15:51:47 2025

@author: Jonna van Mourik
Plot strength of drivers after detrending
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import ttest_ind
import matplotlib.gridspec as gridspec

def load_spei_mask(version, extra=None):
    data_list = []
    for model in models:
        print(model)
        if extra==None:
            path = dir+f"masks/{model}/detrend/mask_{version}_{model}_{reg}_1850-2014.nc"
        else:
            path = dir+f"masks/{model}/detrend/mask_{version}_{model}_{reg}_1850-2014_{extra}.nc"
        da = xr.open_dataarray(path).expand_dims(model=[model])#.resample(time="1MS").mean()
        data_list.append(da)
    return xr.concat(data_list, dim="model", coords="minimal").persist()

#%% Load in data
models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]
reg = "WEU"
reg_title = "Western Europe"
dir = "/your/directory/here/"

print("Load in data var")
if reg=="WEU":
    ts_mrso = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_mrso-local-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_Natl = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-Natl-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_Npac = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-Npac-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_midPac = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-midPac-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_psl_global = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_psl-global-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_psl_local = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_psl-local-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_z500 = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_z500-global-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_z500_local = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_z500-NH-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    name_idx = ["psl-global", "psl-local", "z500-global", "z500-NH", "mrso", "tos-N.Atl", "tos-N.Pac", "tos-Eq.Pac"]#, "pr", "pet"]
    ts_list = [ts_psl_global, ts_psl_local, ts_z500, ts_z500_local, ts_mrso, ts_tos_Natl, ts_tos_Npac, ts_tos_midPac]
elif reg=="SA":
    ts_mrso = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_mrso-local-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_Satl = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-Satl-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_ind = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-ind-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_Npac = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-Npac-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_midPac = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-midPac-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_psl_global = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_psl-global-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_psl_local = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_psl-local-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_z500 = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_z500-global-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    name_idx = ["psl-global", "psl-local", "z500-global", "mrso", "tos-S.Atl", "tos-Ind", "tos-N.Pac", "tos-Eq.Pac"]#, "pr", "pet"]
    ts_list = [ts_psl_global, ts_psl_local, ts_z500, ts_mrso, ts_tos_Satl, ts_tos_ind, ts_tos_Npac, ts_tos_midPac]
    
#Load in everything for SPEI for the region
print("Load in SPEI")
SPEI_reg_MYD_1y = []
SPEI_reg_ND_1y = []
SPEI_list = []
for model in models:
    SPEI_myd = xr.open_dataarray(dir+f"masks/{model}/new_data/detrend/mask_MYD_{model}_{reg}_1850-2014_1yprior.nc").expand_dims(model=[model])
    SPEI_nd = xr.open_dataarray(dir+f"masks/{model}/new_data/detrend/mask_ND_{model}_{reg}_1850-2014_1yprior.nc").expand_dims(model=[model])
    spei_model = xr.open_mfdataset(dir+f"SPEI/{model}/new_data/detrend/SPEI12_monthly_detrend_1850_2014_r*_{model}_1x1.nc", combine="nested", concat_dim="member").__xarray_dataarray_variable__.expand_dims(model=[model])
    SPEI_reg_MYD_1y.append(SPEI_myd)
    SPEI_reg_ND_1y.append(SPEI_nd)
    SPEI_list.append(spei_model)
print("Concat 1y masks")
mask_MYD_1y = xr.concat(SPEI_reg_MYD_1y, dim="model", coords="minimal")
mask_ND_1y = xr.concat(SPEI_reg_ND_1y, dim="model", coords="minimal")
print("Concat normal masks")
mask_MYD = load_spei_mask("MYD")
mask_ND = load_spei_mask("ND")
print("Concat SPEI")
SPEI = xr.concat(SPEI_list, dim="model", coords="minimal")#.persist()
mask_reg = xr.open_dataset(dir+f"masks/1x1/mask_{reg}_1x1.nc").Band1
if reg=="SA":
    SPEI_reg = SPEI.sel(lat=slice(-35, -20), lon=slice(15, 35)).where(mask_reg==1).persist() # for SA
elif reg=="WEU":
    SPEI_reg = SPEI.sel(lat=slice(42,58), lon=slice(-5,17)).where(mask_reg==1).persist() #for WEU


#%% Definitions

def max_values(test_drought, test_index, period, single_model=None):
    """
    Inputs:
        test_drought: timeseries selecting droughts
        test_index: timeseries of climate index
        period: can be either "during", "prior" or "all"
    Outputs:
        prints amount and percentage of hits and misses
        returns the mean value and minimum value of hits (respectively [0] and [1]) and misses (respectively [2] and [3])
        """
    if single_model!=None:
        if period == "preMYD":
            test_drought = mask_MYD_1y.sel(model=single_model).where((mask_MYD_1y.sel(model=single_model)==True) & (mask_MYD.sel(model=single_model)==False), 0)
        elif period == "preND":
            test_drought = mask_ND_1y.sel(model=single_model).where((mask_ND_1y.sel(model=single_model)==True) & (mask_ND.sel(model=single_model)==False), 0)
        elif period == "None":
            test_drought = test_drought.sel(model=single_model)
    else:
        if period == "preMYD":
            test_drought = mask_MYD_1y.where((mask_MYD_1y==True) & (mask_MYD==False), 0)
        elif period == "preND":
            test_drought = mask_ND_1y.where((mask_ND_1y==True) & (mask_ND==False), 0)
        elif period == "None":
            test_drought = test_drought

    #Test if correlation is positive or negative
    if xr.corr(test_index, SPEI_reg.mean(dim=("lat", "lon")))<0:
        test = "positive"
    elif xr.corr(test_index, SPEI_reg.mean(dim=("lat", "lon")))>0:
        test = "negative"
    # Initialize arrays to store hits and misses per member
    hits_max = []

    # Loop over each member and model
    if "model" in test_drought.dims:
        for model in test_drought["model"]:
            for m in test_index["member"]:
                drought_m = test_drought.sel(model=model, member=m).values
                index_m = test_index.sel(model=model, member=m).values
        
                # Identify start and end of ones
                changes = np.diff(np.concatenate(([0], drought_m, [0])))
                start_indices = np.where(changes == 1)[0]
                end_indices = np.where(changes == -1)[0]
                # Count hits and misses
                for start, end in zip(start_indices, end_indices):
                    if test == "negative":
                        hits_max.append(index_m[start:end].min())
                    elif test == "positive":
                        hits_max.append(index_m[start:end].max())
    else: 
        for m in test_index["member"]:
            drought_m = test_drought.sel(member=m).values
            index_m = test_index.sel(model=single_model, member=m).values
    
            # Identify start and end of ones
            changes = np.diff(np.concatenate(([0], drought_m, [0])))
            start_indices = np.where(changes == 1)[0]
            end_indices = np.where(changes == -1)[0]
            # Count hits and misses
            for start, end in zip(start_indices, end_indices):
                if test == "negative":
                    hits_max.append(index_m[start:end].min())
                elif test == "positive":
                    hits_max.append(index_m[start:end].max())

    # Compute total occurrences and percentages
    return hits_max

def significance_marker(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"  # not significant
    
def cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
    return (np.mean(x) - np.mean(y)) / s_pooled

#%% Plot the KDE plots for NDs and MYDs together with all variables in one plot
ts_idx = []
for i in range(len(ts_list)):
    ts_idx.append(ts_list[i].fillna(0))
    
n_vars = len(name_idx)
n_rows = int(np.ceil(n_vars / 2))
x_ticks = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


fig = plt.figure(figsize=(17, 3*n_rows))
ax_topright=None
# Layout: 1+1 (left pair), spacer, 1+1 (right pair)
gs = gridspec.GridSpec(
    n_rows, 5, figure=fig, 
    width_ratios=[1, 1, 0.15, 1, 1],
    wspace=0.08, hspace=0.35         # tighten inside pairs, keep rows spaced
)

fig.suptitle(f"{reg_title}", fontsize=20, y=0.93)
colors = ["red", "purple"]

for i, varname in enumerate(name_idx):
    row = i % n_rows
    col_block = i // n_rows  # 0 = left block, 1 = right block

    # Place subplots in correct block (skip col=2, the spacer)
    ax_before = fig.add_subplot(gs[row, col_block*3])      # col=0 or 3
    ax_during = fig.add_subplot(gs[row, col_block*3 + 1])  # col=1 or 4
    if row == 0 and col_block == 1:
       ax_topright = ax_during

    hits_max_before_MYD = max_values(mask_MYD_1y, ts_idx[i], period="preMYD")
    hits_max_before_ND  = max_values(mask_ND_1y, ts_idx[i], period="preND")
    hits_max_during_MYD = max_values(mask_MYD, ts_idx[i], period="None")
    hits_max_during_ND  = max_values(mask_ND, ts_idx[i], period="None")

    data_before = {"Max, MYD": hits_max_before_MYD, "Max, SYD": hits_max_before_ND}
    data_during = {"Max, MYD": hits_max_during_MYD, "Max, SYD": hits_max_during_ND}
    data_all = [data_before, data_during]

    d_before = cohens_d(hits_max_before_MYD, hits_max_before_ND)
    d_during = cohens_d(hits_max_during_MYD, hits_max_during_ND)

    for j, ax in enumerate([ax_before, ax_during]):
        if i==0 or i==4:
            ax.set_title("Before" if j == 0 else "During", fontsize=14)

        for idx, (lab, data) in enumerate(data_all[j].items()):
            data = np.array(data)
            kde = gaussian_kde(data)
            x_vals = np.linspace(-4, 4, 500)
            y_vals = kde(x_vals)

            ax.fill_between(x_vals, y_vals, alpha=0.3, color=colors[idx])
            mean_val, std_val = np.mean(data), np.std(data)
            mean_y = kde(mean_val)[0]
            ax.plot([mean_val, mean_val], [0, mean_y], color=colors[idx], linestyle='--', linewidth=1.5)
            ax.axvline(x=0, color="grey", linestyle="dashed")

            x_shade = x_vals[(x_vals >= mean_val - std_val) & (x_vals <= mean_val + std_val)]
            y_shade = kde(x_shade)
            ax.fill_between(x_shade, y_shade, alpha=0.2, color=colors[idx], label=lab)

        # effect size annotation
        ax.text(0.05, 0.95, f"d = {d_before:.2f}" if j==0 else f"d = {d_during:.2f}",
                transform=ax.transAxes, fontsize=12, verticalalignment="top")

        ax.set_xlim(-4, 4)
        ax.tick_params(axis='both', labelsize=12)
        # Only set xticks for the lowest row
        if row == n_rows - 1:
            ax.set_xticks(x_ticks)
        else:
            ax.set_xticks([])
    # Sync y-axis and hide duplicate ticks
    ax_before.set_ylim(0, 1)
    ax_during.set_ylim(0, 1)
    ax_during.tick_params(labelleft=False)

    # Add variable name centered above pair
    fig.text(
        (ax_before.get_position().x0 + ax_during.get_position().x1) / 2,
        ax_before.get_position().y1+0.005,
        varname,
        ha="center", va="bottom", fontsize=15)
# Shared labels + legend
fig.supxlabel("Value", fontsize=16, y=0.07)
fig.supylabel("Density", fontsize=16, x=0.08)
handles = [
    plt.Line2D([0], [0], color="red", lw=5, alpha=0.3, label="MYD"),
    plt.Line2D([0], [0], color="purple", lw=5, alpha=0.3, label="SYD")
]
ax_topright.legend(handles=handles,
    fontsize=12,
    loc="upper right",
    frameon=False
)
plt.show()
if reg=="SA":
    fig.savefig("Figure_7.pdf", bbox_inches="tight")
    fig.savefig("Figure_7.jpg", dpi=1200, bbox_inches="tight")
elif reg=="WEU":
    fig.savefig("Figure_8.pdf", bbox_inches="tight")
    fig.savefig("Figure_8.jpg", dpi=1200, bbox_inches="tight")
