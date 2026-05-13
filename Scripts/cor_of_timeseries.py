#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 10:13:34 2025

@author: Jonna van Mourik
Only the correlations of timeseries
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/your/directory/here/")
from scipy.ndimage import label

#definitions
def correlation(var, spei, lags, name):
    correlations = [] 

    for lag in lags:
        shifted_data = spei.shift(time=lag)  # Shift index
        valid = (~np.isnan(var.values)) & (~np.isnan(shifted_data.values))

        if valid.sum() > 0:  # Ensure there are valid values
            cov = np.corrcoef(var.values[valid], shifted_data.values[valid])[0, 1]
        else:
            cov = np.nan, np.nan  # Avoid errors if no valid points

        correlations.append(cov)

    # Convert to numpy arrays
    correlations = np.array(correlations)

    # Find the best lags
    best_lag_cor = lags[np.nanargmax(np.abs(correlations))]    # Max absolute correlation

    print(f"Highest correlation for {name} is {np.max(np.abs(correlations))} at lag: {best_lag_cor}")
    return correlations, best_lag_cor

#%% Now for all models together
models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]
reg = "SA"
dir = "/your/directory/here/"

print("Load in data var")
if reg=="WEU":
    ts_mrso = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_mrso-local-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_global = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-global-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_Natl = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-Natl-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_midAtl = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-midAtl-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_Npac = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-Npac-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_midPac = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-midPac-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_psl_global = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_psl-global-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_psl_local = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_psl-local-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_z500 = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_z500-global-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    #ts_pet = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_PET-mask-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    #ts_pr = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_pr-mask-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_list = [ts_tos_global, ts_tos_Natl, ts_tos_midAtl, ts_tos_midPac, ts_tos_Npac, ts_mrso, ts_psl_global, ts_psl_local, ts_z500]#, ts_pet, ts_pr]
    ts_name = ["tos, global", "tos, Natl", "tos, midAtl", "tos, midPac", "tos, Npac", "mrso", "psl, global", "psl, local", "z500", "pr", "pet"]
    colour = ["#0072B2", "#56B4E9", "blue","tab:cyan", "#92c5de", "pink", "#fe9929", "#F0B400", "#009E73", "#CC79A7", "purple", "violet"]

elif reg=="SA":
    ts_mrso = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_mrso-local-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_global = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-global-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_Satl = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-Satl-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_ind = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-ind-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_Npac = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-Npac-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_tos_midPac = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_tos-midPac-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_psl_global = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_psl-global-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_psl_local = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_psl-local-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_z500 = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_z500-global-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    #ts_pet = xr.open_dataarray(dir+f"SVD_output/detrend/ts0_PET-mask-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    #ts_pr = xr.open_dataarray(dir+f"SVD_output/detrend/ts0_pr-mask-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
    ts_list = [ts_tos_global, ts_tos_Satl, ts_tos_ind, ts_tos_midPac, ts_tos_Npac, ts_mrso, ts_psl_global, ts_psl_local, ts_z500]#, ts_pr, ts_pet]
    ts_name = ["tos, global", "tos, Satl", "tos, ind", "tos, midPac", "tos, Npac", "mrso", "psl, global", "psl, local", "z500", "pr", "pet"]
    colour = ["#0072B2", "#56B4E9", "blue","tab:cyan", "#92c5de", "pink", "#fe9929", "#F0B400", "#009E73", "#CC79A7", "purple", "violet"]

print("Load in data SPEI")
SPEI_list = []
for model in models:
    spei_model = xr.open_mfdataset(dir+f"SPEI/{model}/detrend/SPEI12_monthly_detrend_1850_2014_r*_{model}_{reg}.nc", combine="nested", concat_dim="member").__xarray_dataarray_variable__.expand_dims(model=[model])

    SPEI_list.append(spei_model)
SPEI = xr.concat(SPEI_list, dim="model", coords="minimal")

lags = np.arange(-18, 13)  # from -18 to +12

#%% White noise band per region


def compute_multi_model_threshold(ts_list, SPEI, lags, q=0.975):
    """
    Computes a multi-model, multi-variable white noise threshold for correlations.
    
    Parameters
    ----------
    ts_list : list of xarray.DataArray
        List of time series (model, member, time)
    SPEI : xarray.DataArray
        SPEI time series (model, member, time)
    lags : array
        Array of lag values to test
    q : float
        Quantile for threshold (default=0.975 for two-sided 95%)
    """
    correlations_all = []

    for var_da in ts_list:
        for model in var_da.model:
            print(model)
            spei_m = SPEI.sel(model=model)
            var_m = var_da.sel(model=model)

            nmem = spei_m.sizes['member']
            pairs = [(i, j) for i in range(nmem) for j in range(nmem) if i != j]

            for (spei_idx, var_idx) in pairs:
                cors = []
                spei_ts = spei_m.sel(member=spei_idx)
                var_ts = var_m.sel(member=var_idx)

                for lag in lags:
                    shifted = spei_ts.shift(time=lag)
                    valid = (~np.isnan(var_ts.values)) & (~np.isnan(shifted.values))

                    if valid.sum() > 1:
                        r = np.corrcoef(var_ts.values[valid], shifted.values[valid])[0, 1]
                        cors.append(abs(r))
                    else:
                        cors.append(np.nan)

                correlations_all.append(cors)

    # Convert to DataArray for pooling across variables and models
    print("Calculate noise_da")
    noise_da = xr.DataArray(data=np.array(correlations_all), dims=["combination", "lag"], coords={"combination": np.arange(len(correlations_all)), "lag": lags}, name="white_noise_correlations")

    # Quantile threshold across all combinations
    threshold_da = noise_da.quantile(q, dim="combination")
    return threshold_da

threshold_da = compute_multi_model_threshold(ts_list, SPEI, lags, q=0.975)
threshold_da.to_netcdf(dir+f"SVD_output/new_data/detrend/white_noise_MMM_allvar_{reg}.nc")
threshold_da.plot()
plt.ylim(0, 0.8)

#%% Make version with both South Africa and Western Europe
#Western Europe
ts_mrso_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_mrso-local-mode0_WEU_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_tos_global_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-global-mode0_WEU_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_tos_Natl_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-Natl-mode0_WEU_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_tos_Npac_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-Npac-mode0_WEU_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_tos_midPac_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-midPac-mode0_WEU_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_psl_global_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_psl-global-mode0_WEU_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_psl_local_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_psl-local-mode0_WEU_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_z500_global_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_z500-global-mode0_WEU_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_z500_local_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_z500-NH-mode0_WEU_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
#ts_pet = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_PET-mask-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
#ts_pr = xr.open_dataarray(dir+f"SVD_output/new_data/detrend/ts0_pr-mask-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_list_WEU = [ts_tos_global_WEU, ts_tos_Natl_WEU, ts_tos_midPac_WEU, ts_tos_Npac_WEU, ts_mrso_WEU, ts_psl_global_WEU, ts_psl_local_WEU, ts_z500_local_WEU, ts_z500_global_WEU]#, ts_pet, ts_pr]
ts_name_WEU = ["tos, global", "tos, Natl", "tos, EqPac", "tos, Npac", "mrso", "psl, global", "psl, local", "z500, local", "z500, global"]
colours_WEU = ["blue", "#084594","dodgerblue", "#41b6c4", "orchid", "#fe9929", "#F0B400", "indianred", "firebrick"]
#South Africa
ts_mrso_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_mrso-local-mode0_SA_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_tos_global_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-global-mode0_SA_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_tos_Satl_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-Satl-mode0_SA_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_tos_ind_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-ind-mode0_SA_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_tos_Npac_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-Npac-mode0_SA_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_tos_midPac_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-midPac-mode0_SA_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_psl_global_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_psl-global-mode0_SA_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_psl_local_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_psl-local-mode0_SA_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_z500_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_z500-global-mode0_SA_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
#ts_pet = xr.open_dataarray(dir+f"SVD_output/detrend/ts0_PET-mask-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
#ts_pr = xr.open_dataarray(dir+f"SVD_output/detrend/ts0_pr-mask-mode0_{reg}_MMM_historical.nc").assign_coords(model=("model", models)).sel(mode=0)
ts_list_SA = [ts_tos_global_SA, ts_tos_Satl_SA, ts_tos_ind_SA, ts_tos_midPac_SA, ts_tos_Npac_SA, ts_mrso_SA, ts_psl_global_SA, ts_psl_local_SA, ts_z500_SA]#, ts_pr, ts_pet]
ts_name_SA = ["tos, global", "tos, Satl", "tos, ind", "tos, EqPac", "tos, Npac", "mrso", "psl, global", "psl, local", "z500, global"]
colours_SA = ["blue", "#084594", "cyan","dodgerblue", "#41b6c4", "orchid", "#fe9929", "#F0B400", "firebrick"]

print("Load in data SPEI")
SPEI_list_WEU = []
SPEI_list_SA = []
for model in models:
    spei_model_WEU = xr.open_mfdataset(dir+f"SPEI/{model}/new_data/detrend/SPEI12_monthly_detrend_1850_2014_r*_{model}_WEU.nc", combine="nested", concat_dim="member").__xarray_dataarray_variable__.expand_dims(model=[model])
    spei_model_SA = xr.open_mfdataset(dir+f"SPEI/{model}/new_data/detrend/SPEI12_monthly_detrend_1850_2014_r*_{model}_SA.nc", combine="nested", concat_dim="member").__xarray_dataarray_variable__.expand_dims(model=[model])

    SPEI_list_WEU.append(spei_model_WEU)
    SPEI_list_SA.append(spei_model_SA)
SPEI_WEU = xr.concat(SPEI_list_WEU, dim="model", coords="minimal")
SPEI_SA = xr.concat(SPEI_list_SA, dim="model", coords="minimal")
threshold_da_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/white_noise_MMM_allvar_SA.nc")
threshold_da_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/white_noise_MMM_allvar_WEU.nc")

lower = threshold_da_WEU*0
upper_SA = threshold_da_SA.mean()
upper_WEU = threshold_da_WEU.mean()

#Plot
fig, ax = plt.subplots(1,2, figsize=(13, 4), sharey=True)
plt.subplots_adjust(wspace=0.05)

for i in range(len(ts_list_SA)):
    correlations, best_lag_cor = correlation(ts_list_SA[i], SPEI_SA, lags, ts_name_SA[i])
    #  Plot the Results
    ax[0].plot(lags, np.abs(correlations), label=ts_name_SA[i], color=colours_SA[i])  # Normalize for scale
    ax[0].scatter(x=best_lag_cor, y=np.abs(correlations[np.where(lags==best_lag_cor)]), color=colours_SA[i], zorder=10)
    print(i)
for i in range(len(ts_list_WEU)):
    correlations, best_lag_cor = correlation(ts_list_WEU[i], SPEI_WEU, lags, ts_name_WEU[i])
    #  Plot the Results
    ax[1].plot(lags, np.abs(correlations), label=ts_name_WEU[i], color=colours_WEU[i])  # Normalize for scale
    ax[1].scatter(x=best_lag_cor, y=np.abs(correlations[np.where(lags==best_lag_cor)]), color=colours_WEU[i], zorder=10)
    print(i)
    
ax[0].fill_between(lags, lower, upper_SA, color="grey", alpha=0.3)
ax[1].fill_between(lags, lower, upper_WEU, color="grey", alpha=0.3)

for axes in ax.flatten():
    axes.set_ylim(0, 0.8)
    axes.set_xlim(-18, 12)
    axes.grid()
    axes.legend()

fig.supxlabel("Lag (months)")
ax[0].set_ylabel(r"Correlation $r$ (-)")
ax[0].set_title("a)", loc="left")
ax[0].set_title("Southern Africa")
ax[1].set_title("b)", loc="left")
ax[1].set_title("Western Europe")

fig.savefig("Figure_5.pdf", bbox_inches="tight")
fig.savefig("Figure_5.jpg", bbox_inches="tight", dpi=1200)
plt.show()

#%% Do the same thing for PET and PR
models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]

print("Load in data SPEI")
SPEI_list_WEU = []
SPEI_list_SA = []
for model in models:
    spei_model_WEU = xr.open_mfdataset(dir+f"SPEI/{model}/new_data/detrend/SPEI12_monthly_detrend_1850_2014_r*_{model}_WEU.nc", combine="nested", concat_dim="member").__xarray_dataarray_variable__.expand_dims(model=[model])
    spei_model_SA = xr.open_mfdataset(dir+f"SPEI/{model}/new_data/detrend/SPEI12_monthly_detrend_1850_2014_r*_{model}_SA.nc", combine="nested", concat_dim="member").__xarray_dataarray_variable__.expand_dims(model=[model])

    SPEI_list_WEU.append(spei_model_WEU)
    SPEI_list_SA.append(spei_model_SA)
SPEI_WEU = xr.concat(SPEI_list_WEU, dim="model", coords="minimal")
SPEI_SA = xr.concat(SPEI_list_SA, dim="model", coords="minimal")

threshold_da_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/white_noise_MMM_allvar_SA.nc")
threshold_da_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/white_noise_MMM_allvar_WEU.nc")

lower = threshold_da_WEU*0
upper_SA = threshold_da_SA.mean()
upper_WEU = threshold_da_WEU.mean()
lags = np.arange(-18, 13)  # from -18 to +12

PET12_WEU = xr.open_dataarray(dir+"PET12-WEU_MM.nc")
PET12_SA = xr.open_dataarray(dir+"PET12-SA_MM.nc")
PR12_WEU = xr.open_dataarray(dir+"PR12-WEU_MM.nc")
PR12_SA = xr.open_dataarray(dir+"PR12-SA_MM.nc")

ts_name = ["PET-12", "PR-12"] 
ts_list_SA = [PET12_SA, PR12_SA] 
ts_list_WEU = [PET12_WEU, PR12_WEU] 

colours = ["indianred", "royalblue", "indianred", "royalblue"]
linestyle = ["-", "-", "--", "--"]

#definition for perfect tent
def tent(x, y0):
    return np.where(x < 0, y0/12 * (x + 12), -y0/12 * (x - 12))

fig, ax = plt.subplots(1,2, figsize=(13, 4), sharey=True)
plt.subplots_adjust(wspace=0.05)

for i in range(len(ts_list_SA)):
    correlations, best_lag_cor = correlation(ts_list_SA[i], SPEI_SA, lags, ts_name_SA[i])
    #  Plot the Results
    ax[0].plot(lags, np.abs(correlations), label=ts_name[i], color=colours[i], linestyle=linestyle[i])  # Normalize for scale
    ax[0].scatter(x=best_lag_cor, y=np.abs(correlations[np.where(lags==best_lag_cor)]), color=colours[i], zorder=10)
    print(i)
    if i==0:
        ax[0].plot(np.arange(-12, 13), tent(np.arange(-12, 13), np.abs(correlations[np.where(lags==best_lag_cor)])), color="black", linestyle="dashed", alpha=0.5)
    elif i==1:
        ax[0].plot(np.arange(-12, 13), tent(np.arange(-12, 13), np.abs(correlations[np.where(lags==best_lag_cor)])), color="black", linestyle="dashed", alpha=0.5, label="Unimpacted-12")

for i in range(len(ts_list_WEU)):
    correlations, best_lag_cor = correlation(ts_list_WEU[i], SPEI_WEU, lags, ts_name_WEU[i])
    #  Plot the Results
    ax[1].plot(lags, np.abs(correlations), label=ts_name[i], color=colours[i], linestyle=linestyle[i])  # Normalize for scale
    ax[1].scatter(x=best_lag_cor, y=np.abs(correlations[np.where(lags==best_lag_cor)]), color=colours[i], zorder=10)
    print(i)
    if i==0:
        ax[1].plot(np.arange(-12, 13), tent(np.arange(-12, 13), np.abs(correlations[np.where(lags==best_lag_cor)])), color="black", linestyle="dashed", alpha=0.5)
    elif i==1:    
        ax[1].plot(np.arange(-12, 13), tent(np.arange(-12, 13), np.abs(correlations[np.where(lags==best_lag_cor)])), color="black", linestyle="dashed", alpha=0.5, label="Unimpacted-12")
    
ax[0].fill_between(lags, lower, upper_SA, color="grey", alpha=0.3)
ax[1].fill_between(lags, lower, upper_WEU, color="grey", alpha=0.3)

for axes in ax.flatten():
    axes.set_ylim(0, 1)
    axes.set_xlim(-18, 12)
    axes.grid()
    axes.legend()

fig.supxlabel("Lag (months)")
ax[0].set_ylabel(r"Correlation $r$ (-)")
ax[0].set_title("a)", loc="left")
ax[0].set_title("Southern Africa")
ax[1].set_title("b)", loc="left")
ax[1].set_title("Western Europe")

fig.savefig("Figure_6.pdf", bbox_inches="tight")
fig.savefig("Figure_6.jpg", bbox_inches="tight", dpi=1200)
plt.show()

#%% Also make a figure on the correlations with all the variables wrt PET and PR

lower = threshold_da_WEU*0

#First for PET
fig, ax = plt.subplots(1,2, figsize=(13, 4), sharey=True)
plt.subplots_adjust(wspace=0.05)

for i in range(len(ts_list_SA)):
    correlations, best_lag_cor = correlation(ts_list_SA[i], PET12_SA, lags, ts_name_SA[i])
    #  Plot the Results
    ax[0].plot(lags, np.abs(correlations), label=ts_name_SA[i], color=colours_SA[i])  # Normalize for scale
    ax[0].scatter(x=best_lag_cor, y=np.abs(correlations[np.where(lags==best_lag_cor)]), color=colours_SA[i], zorder=10)
    print(i)
for i in range(len(ts_list_WEU)):
    correlations, best_lag_cor = correlation(ts_list_WEU[i], PET12_WEU, lags, ts_name_WEU[i])
    #  Plot the Results
    ax[1].plot(lags, np.abs(correlations), label=ts_name_WEU[i], color=colours_WEU[i])  # Normalize for scale
    ax[1].scatter(x=best_lag_cor, y=np.abs(correlations[np.where(lags==best_lag_cor)]), color=colours_WEU[i], zorder=10)
    print(i)
    
ax[0].fill_between(lags, lower, upper_SA, color="grey", alpha=0.3)
ax[1].fill_between(lags, lower, upper_WEU, color="grey", alpha=0.3)

for axes in ax.flatten():
    axes.set_ylim(0, 0.8)
    axes.set_xlim(-18, 12)
    axes.grid()
    axes.legend()

fig.supxlabel("Lag (months)")
fig.suptitle("Correlations with respect to PET-12")
ax[0].set_ylabel(r"Correlation $r$ (-)")
ax[0].set_title("a)", loc="left")
ax[0].set_title("South Africa")
ax[1].set_title("b)", loc="left")
ax[1].set_title("Western Europe")

fig.savefig("Figure_S6.pdf", bbox_inches="tight")
fig.savefig("Figure_S6.jpg", bbox_inches="tight", dpi=1200)
plt.show()

#%% And precipitation
threshold_pr_SA = xr.open_dataarray(dir+"SVD_output/detrend/white_noise_MMM_allvar-pr_SA.nc")
threshold_pr_WEU = xr.open_dataarray(dir+"SVD_output/detrend/white_noise_MMM_allvar-pr_WEU.nc")

lower = threshold_da_WEU*0
upper_pr_SA = threshold_pr_SA.mean()
upper_pr_WEU = threshold_pr_WEU.mean()

fig, ax = plt.subplots(1,2, figsize=(13, 4), sharey=True)
plt.subplots_adjust(wspace=0.05)

for i in range(len(ts_list_SA)):
    correlations, best_lag_cor = correlation(ts_list_SA[i], PR12_SA, lags)
    #  Plot the Results
    ax[0].plot(lags, np.abs(correlations), label=ts_name_SA[i], color=colours_SA[i])  # Normalize for scale
    ax[0].scatter(x=best_lag_cor, y=np.abs(correlations[np.where(lags==best_lag_cor)]), color=colours_SA[i], zorder=10)
    print(i)
for i in range(len(ts_list_WEU)):
    correlations, best_lag_cor = correlation(ts_list_WEU[i], PR12_WEU, lags)
    #  Plot the Results
    ax[1].plot(lags, np.abs(correlations), label=ts_name_WEU[i], color=colours_WEU[i])  # Normalize for scale
    ax[1].scatter(x=best_lag_cor, y=np.abs(correlations[np.where(lags==best_lag_cor)]), color=colours_WEU[i], zorder=10)
    print(i)
    
ax[0].fill_between(lags, lower, upper_pr_SA, color="grey", alpha=0.3)
ax[1].fill_between(lags, lower, upper_pr_WEU, color="grey", alpha=0.3)

for axes in ax.flatten():
    axes.set_ylim(0, 0.8)
    axes.set_xlim(-18, 12)
    axes.grid()
    axes.legend()

fig.supxlabel("Lag (months)")
fig.suptitle("Correlations with respect to PR-12")
ax[0].set_ylabel(r"Correlation $r$ (-)")
ax[0].set_title("a)", loc="left")
ax[0].set_title("South Africa")
ax[1].set_title("b)", loc="left")
ax[1].set_title("Western Europe")
fig.savefig("Figure_S7.pdf", bbox_inches="tight")
fig.savefig("Figure_S7.jpg", bbox_inches="tight", dpi=1200)
plt.show()
