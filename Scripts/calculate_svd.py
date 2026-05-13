#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:17:44 2025

@author: Jonna van Mourik
Simple version of the SVD analysis with only the necessary figures
"""
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.utils.extmath import randomized_svd
import sys
sys.path.append("/your/directory/here/")
import dask.array as da
import matplotlib.pyplot as plt

# Definitions
def lagged_svd_mm(data1, data2, lag):
    # Replace NaNs with zeros
    print("replace nans", flush=True)
    data1_filled = data1.fillna(0)
    data2_filled = data2.fillna(0)
    print("stack dimensions", flush=True)
    # Stack spatial dimensions and concatenate all members
    data1_flat = data1_filled.stack(spatial=("lat", "lon"))
    data2_flat = data2_filled.stack(spatial=("lat", "lon"))
    print("Reshape", flush=True)
    # Reshape to merge members and time: (model, member, time, spatial) → (total_time, spatial)
    if "model" in data1.dims:
        data1_flat_merged = data1_flat.stack(sample=("model", "member", "time")).transpose("sample", "spatial")#.values
        data2_flat_merged = data2_flat.stack(sample=("model", "member", "time")).transpose("sample", "spatial")#.values
    else:
        data1_flat_merged = data1_flat.stack(sample=("member", "time")).transpose("sample", "spatial")#.values
        data2_flat_merged = data2_flat.stack(sample=("member", "time")).transpose("sample", "spatial")#.values
    print("Apply lag", flush=True)
    # Apply lag
    if lag < 0:
        lagged_data1 = data1_flat_merged[:lag]
        lagged_data2 = data2_flat_merged[-lag:]
    elif lag > 0:
        lagged_data1 = data1_flat_merged[lag:]
        lagged_data2 = data2_flat_merged[:-lag]
    else:
        lagged_data1 = data1_flat_merged
        lagged_data2 = data2_flat_merged
        
    print("Chunk", flush=True)
    lagged_data1 = lagged_data1.chunk({"spatial": 5000})  # adjust chunk size to fit memory
    lagged_data2 = lagged_data2.chunk({"spatial": 5000})
    print("Ckeck false infinities", flush=True)
    data1_lagged_clean = lagged_data1.where(np.isfinite(lagged_data1), 0.0)
    data2_lagged_clean = lagged_data2.where(np.isfinite(lagged_data2), 0.0)
    print("Get values", flush=True)
    X = data1_lagged_clean.astype(np.float32).data  # returns a dask array
    Y = data2_lagged_clean.astype(np.float32).data
    N_x = data1_flat.sizes["spatial"]
    N_y = data2_flat.sizes["spatial"]
    # Compute cross-covariance matrix
    print("Compute cross-covariance matrix", flush=True)
    C = da.matmul(X.T, Y) / np.sqrt(N_x * N_y)
    C_np = C.compute()  # Only compute once here

    # Perform SVD
    print("perform SVD", flush=True)
    n_modes=1
    u, s, vt = randomized_svd(C_np, n_components=n_modes, n_iter=5)  # Faster randomized SVD
    #Also calculate timeseries
    # Compute amplitudes for the first n_modes
    print("Calculate amplitudes")
    amplitudes1 = X @ u[:, :n_modes]
    amplitudes2 = Y @ vt[:n_modes, :].T
    print("Calculate timeseries")
    timeseries1 = amplitudes1/np.std(amplitudes1)
    timeseries2 = amplitudes2/np.std(amplitudes2)
    time = np.arange(0, len(amplitudes1), 1)
    modes = np.arange(0, n_modes)
    return u, s, vt, xr.DataArray(timeseries1, dims=("time", "mode"), coords={"time": time, "mode": modes}, name="timeseries"), xr.DataArray(timeseries2, dims=("time", "mode"), coords={"time": time, "mode": modes}, name="timeseries")

def create_da(data, lat, lon, name):
    return xr.DataArray(data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, name=name)

def compute_scf(singular_values):
    """Compute Squared Covariance Fraction (SCF) for each mode."""
    squared_singular_values = singular_values**2
    total = np.sum(squared_singular_values)
    scf = (squared_singular_values / total)*100
    return scf
    
def compute_mode_variance_contributions(lagged_data1, lagged_data2, U, Vt, num_modes=3):
    """
    Compute the fraction of variance explained in each original field by each SVD mode.
    """
    # Project original data onto left and right singular vectors
    # Each mode's time series: shape (mode, time)
    A = lagged_data1 @ U[:, :num_modes]        # (T, X) x (X, num_modes) = (T, num_modes)
    B = lagged_data2 @ Vt[:num_modes, :].T     # (T, Y) x (Y, num_modes) = (T, num_modes)

    # Compute variance in each mode
    var_A = np.sum(A**2, axis=0)               # shape (num_modes,)
    var_B = np.sum(B**2, axis=0)

    # Compute total variance in each field
    total_var1 = np.sum(lagged_data1**2)
    total_var2 = np.sum(lagged_data2**2)

    # Fraction of variance explained by each mode in each field
    C_p = var_A / total_var1
    C_q = var_B / total_var2

    correlations = np.array([
        np.corrcoef(A[:, i], B[:, i])[0, 1]
        for i in range(num_modes)
    ])

    return C_p, C_q, correlations

def compute_mode_variance_contributions_dask(lagged_data1, lagged_data2, U, Vt, num_modes=10):
    """
    Compute the fraction of variance explained in each original field by each SVD mode,
    using Dask to avoid loading full arrays into memory.
    """
    # Project onto modes (U, Vt should be small enough to be in memory)
    A = da.matmul(lagged_data1, U[:, :num_modes])           # shape (T, num_modes)
    B = da.matmul(lagged_data2, Vt[:num_modes, :].T)         # shape (T, num_modes)

    # Variance per mode
    var_A = da.sum(A ** 2, axis=0)
    var_B = da.sum(B ** 2, axis=0)

    # Total variance
    total_var1 = da.sum(lagged_data1 ** 2)
    total_var2 = da.sum(lagged_data2 ** 2)

    # Fraction of variance explained
    C_p = var_A / total_var1
    C_q = var_B / total_var2

    # Correlations (force compute small arrays)
    A_local = A.compute()
    B_local = B.compute()
    correlations = np.array([
        np.corrcoef(A_local[:, i], B_local[:, i])[0, 1]
        for i in range(num_modes)
    ])

    return C_p.compute(), C_q.compute(), correlations

def load_variable(varname):
    data_list = []
    for model in models:
        print(model, flush=True)
        if varname=="PET":
            path = dir+f"PET/{model}/new_data/detrend/std_anom_pet_mon-detrend_{model}_historical_1850-2014_1x1.nc"
        else:
            path = dir+f"CMIP6/{model}/detrend/std_anom_{varname}_mon-detrend_{model}_historical_1850-2014_1x1.nc"
        da = xr.open_dataarray(path).expand_dims(model=[model])
        data_list.append(da)
    return xr.concat(data_list, dim="model", coords="minimal", compat="override")#.compute()

def save_single_mode_svd(u_da, vt_da, output_dir, reg, mode=0, name1="var", name2="SPEI"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    for lag in u_da:
        ds = xr.Dataset({
            f"U_mode{mode}_{name1}": u_da[lag],
            f"V_mode{mode}_{name2}": vt_da[lag]
        })
        filename = f"SVD_mode{mode}_lag{lag:+d}_{name1}_{name2}_{reg}.nc"
        ds.to_netcdf(os.path.join(output_dir, filename))
        print(f"Saved {filename}")

def reshape(da, models):
    print("Squeeze", flush=True)
    original_data = da.data.squeeze()
    members = np.arange(10)
    models = models
    time = var_stdanom.time
    print("Calculate timeseries", flush=True)
    if "mode" in da.dims:
        reshaped_data = original_data.reshape(6, 10,1980,len(da.mode))
        new_timeseries = xr.DataArray(
            reshaped_data,
            dims=("model", "member", "time", "mode"),
            coords={"model": models, "member": members, "time": time, "mode":da.mode}, name="timeseries")
    else:
        reshaped_data = original_data.reshape(6, 10, 1980)
        new_timeseries = xr.DataArray(
            reshaped_data,
            dims=("model", "member", "time"),
            coords={"model":models, "member": members, "time": time}, name="timeseries")
    return new_timeseries

#%% 
models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]
reg = "SA"
dir = "/archive/depfg/6196306/"
var = "tos"
region = "Satl" #or atl ind or pac or global or anything else
lags = [-18, -12, -6, 0, 6]

#%% For all models combined
print("Load in data var", flush=True)
var_stdanom = load_variable(var).sel(time=slice("1850", "2014"))
#Load in everything for SPEI for the region
print("Load in data SPEI", flush=True)
SPEI_list = []
for model in models:
    spei_model = xr.open_mfdataset(dir+f"SPEI/{model}/new_data/detrend/SPEI12_monthly_detrend_1850_2014_r*_{model}_1x1.nc", combine="nested", concat_dim="member").__xarray_dataarray_variable__.expand_dims(model=[model])
    SPEI_list.append(spei_model)
SPEI = xr.concat(SPEI_list, dim="model", coords="minimal")

mask_reg = xr.open_dataset(dir+f"masks/1x1/mask_{reg}_1x1.nc").Band1


print("Calculate SVD", flush=True)
if reg=="SA": 
    SPEI_reg = SPEI.sel(lat=slice(-35, -10), lon=slice(12, 37)).where(mask_reg==1)#.compute() # for SA
    if region=="local":
        data1 = var_stdanom.sel(lat=slice(-35, -10), lon=slice(12,37))
    elif region=="mask":
        data1 = var_stdanom.sel(lat=slice(-35, -10), lon=slice(12, 37)).where(mask_reg==1)
    elif var=="tos" and region=="midPac":
        data1 = var_stdanom.sel(lat=slice(-20, 20), lon=slice(-180, -80))
    elif var=="tos" and region=="Satl": 
        data1 = var_stdanom.sel(lat=slice(-60, 20), lon=slice(-40, 20))
    elif var=="tos" and region=="ind":
        data1 = var_stdanom.sel(lat=slice(-40,20), lon=slice(40, 140))
    elif var=="tos" and region=="Npac":
        data1 = var_stdanom.assign_coords(lon=(var_stdanom.lon % 360)).sortby("lon").copy(deep=True).sel(lat=slice(20,60), lon=slice(130,240))
    elif region=="global":
        data1 = var_stdanom
    elif region=="Satl_small":
        data1 = var_stdanom.sel(lat=slice(-28, -10), lon=slice(2, 15))
    elif region=="ATL3":
        data1 = var_stdanom.sel(lat=slice(-3, 3), lon=slice(-20, 0))
    elif region=="CABA":
        data1 = var_stdanom.sel(lat=slice(-20, -10), lon=slice(2, 4))
    elif region=="MEQUE":
        data1 = var_stdanom.sel(lat=slice(-28, -10), lon=slice(8, 15))
        
elif reg=="WEU":
    SPEI_reg = SPEI.where(mask_reg==1).sel(lat=slice(42,58), lon=slice(-5,17))
    if var=="psl" and region=="local":
        data1 = var_stdanom.sel(lat=slice(20,80), lon=slice(-80,40))
    elif region=="mask":
        data1 = var_stdanom.sel(lat=slice(42, 58), lon=slice(-5, 17)).where(mask_reg==1)
    elif region=="global":
        data1 = var_stdanom
    elif var=="tos" and region=="Natl":
        data1 = var_stdanom.sel(lat=slice(20,80), lon=slice(-80,40))
    elif var=="tos" and region=="midPac":
        data1 = var_stdanom.sel(lat=slice(-20, 20), lon=slice(-180, -80))
    elif var=="tos" and region=="midAtl":
        data1 = var_stdanom.sel(lat=slice(-20,20), lon=slice(-50,10))
    elif var=="tos" and region=="Npac":
        data1 = var_stdanom.assign_coords(lon=(var_stdanom.lon % 360)).sortby("lon").copy(deep=True).sel(lat=slice(20,60), lon=slice(130,240))
    elif var=="mrso" and region=="local": 
        data1 = var_stdanom.sel(lat=slice(42,58), lon=slice(-5,17))
    elif region=="NH":
        data1 = var_stdanom.sel(lat=slice(30, 65))
        
data2 = SPEI_reg
data2_name = "SPEI"
mode = 0

svd = {lag: lagged_svd_mm(data1, data2, lag) for lag in lags}
sys.stdout.flush()


#Test which mode is the same as the SPEI figures
print("Save SVD", flush=True)
u_mode = {lag: u[:, mode].reshape((data1.lat.size, data1.lon.size)) for lag, (u, _, _, _, _) in svd.items()}
vt_mode = {lag: vt[mode, :].reshape((data2.lat.size, data2.lon.size)) for lag, (_, _, vt, _, _) in svd.items()}
# Convert to xarray DataArrays
u_da = {lag: create_da(u, data1.lat, data1.lon, f"U Mode {mode}") for lag, u in u_mode.items()}
vt_da = {lag: create_da(vt, data2.lat, data2.lon, f"V Mode {mode}") for lag, vt in vt_mode.items()}

save_single_mode_svd(u_da, vt_da, output_dir=dir+"SVD_output/new_data/detrend/", reg=reg, mode=0, name1=var+"_"+region, name2="SPEI")

print("Calculate timeseries", flush=True)
ts_var0 = svd[0][3] 
ts_var1 = svd[0][4] 
print("Reshape and save timeseries", flush=True)
#Reshape to model,member,time
ts_rs_var0 = reshape(ts_var0, models).to_netcdf(dir+f"SVD_output/new_data/detrend/ts0_{var}-{region}-mode0_{reg}_MMM_historical.nc")
ts_rs_var1 = reshape(ts_var1, models).to_netcdf(dir+f"SVD_output/new_data/detrend/ts1_{var}-{region}-mode0_{reg}_MMM_historical.nc")

#%% Make the plots
print("Now make the plots")
for lag in lags:
    u_da = xr.open_dataset(dir+f"SVD_output/new_data/detrend/SVD_mode0_lag{lag:+d}_{var}_{region}_SPEI_{reg}.nc")[f"U_mode0_{var}_{region}"]
    vt_da = xr.open_dataset(dir+f"SVD_output/new_data/detrend/SVD_mode0_lag{lag:+d}_{var}_{region}_SPEI_{reg}.nc").V_mode0_SPEI
    if var=="mrso":
        cmap = "BrBG"#"BrBG"
    else: 
        cmap = "coolwarm"#"BrBG"
    # Plot and save figures
    fig_width = 6
    # Get aspect ratio of region (latitude / longitude span)
    lat_min = float(data1.lat.min())
    lat_max = float(data1.lat.max())
    lon_min = float(data1.lon.min())
    lon_max = float(data1.lon.max())
    
    aspect_ratio = (lat_max - lat_min) / (lon_max - lon_min)
    fig_height = fig_width * aspect_ratio  # dynamic height based on region
    
    fig, ax = plt.subplots(1,1, figsize=(fig_width, fig_height), subplot_kw={"projection": ccrs.PlateCarree()})
    #put u_da[lag] back if for more lags
    u_da.plot(ax=ax, cmap=cmap, transform=ccrs.PlateCarree(), vmin=-0.03, vmax=0.03, cbar_kwargs={
    "orientation": "vertical",
    "shrink": 0.62,
    "pad": 0.05,
    "label": " ",
    "format": '%.2f'})
    ax.coastlines()
    ax.contour(mask_reg.lon, mask_reg.lat, np.isnan(mask_reg), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_title(f"{var}, lag {lag}, MPI-ESM1-2-LR")
    filename = f"/eejit/home/6196306/Data/Figures/Remake_ch3/Lagged_SVD_{reg}_{var}_{region}_lag={lag}_{data2_name}-pattern_mode={mode}_MPI-ESM1-2-LR_v3"
    fig.savefig(f"{filename}.jpg", dpi=1200, bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig, ax = plt.subplots(1,1, figsize=(fig_width, fig_height), subplot_kw={"projection": ccrs.PlateCarree()})
    (u_da*-1).plot(ax=ax, cmap=cmap, transform=ccrs.PlateCarree(), vmin=-0.03, vmax=0.03, cbar_kwargs={
    "orientation": "vertical",
    "shrink": 0.62,
    "pad": 0.05,
    "label": " ",
    "format": '%.2f'})
    ax.coastlines()
    ax.contour(mask_reg.lon, mask_reg.lat, np.isnan(mask_reg), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_title(f"{var}, lag {lag}, {model}")
    filename = f"/eejit/home/6196306/Data/Figures/Remake_ch3/Lagged_SVD_{reg}_{var}_{region}_lag={lag}_{data2_name}-pattern_mode={mode}_MPI-ESM1-2-LR_reverse_v3"
    fig.savefig(f"{filename}.jpg", dpi=1200, bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    
    #Now plot the same but for SPEI
    fig, ax = plt.subplots(1,1, figsize=(fig_width, fig_height), subplot_kw={"projection": ccrs.PlateCarree()})
    vt_da.plot(ax=ax, cmap=cmap, transform=ccrs.PlateCarree(), vmin=-0.03, vmax=0.03, cbar_kwargs={
    "orientation": "vertical",
    "shrink": 0.62,
    "pad": 0.05,
    "label": " ",
    "format": '%.2f'})
    ax.coastlines()
    ax.contour(mask_reg.lon, mask_reg.lat, np.isnan(mask_reg), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    if reg=="WEU":
        ax.set_extent([-80, 40, 20, 80], crs=ccrs.PlateCarree())
    elif reg=="SA":
        ax.set_extent([12, 37, -35, -10], crs=ccrs.PlateCarree())
    ax.set_title(f"SPEI, lag {lag}, {model}")
    filename = f"/eejit/home/6196306/Data/Figures/Remake_ch3/Lagged_SVD_{reg}_{var}_{region}_lag={lag}_{data2_name}_vt-pattern_mode={mode}_{model}_v3"
    fig.savefig(f"{filename}.jpg", dpi=1200, bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig, ax = plt.subplots(1,1, figsize=(fig_width, fig_height), subplot_kw={"projection": ccrs.PlateCarree()})
    (vt_da*-1).plot(ax=ax, cmap=cmap, transform=ccrs.PlateCarree(), vmin=-0.03, vmax=0.03, cbar_kwargs={
    "orientation": "vertical",
    "shrink": 0.62,
    "pad": 0.05,
    "label": " ",
    "format": '%.2f'})
    ax.coastlines()
    ax.contour(mask_reg.lon, mask_reg.lat, np.isnan(mask_reg), colors='black', linewidths=2, transform=ccrs.PlateCarree())
    if reg=="WEU":
        ax.set_extent([-80, 40, 20, 80], crs=ccrs.PlateCarree())
    elif reg=="SA":
        ax.set_extent([12, 37, -35, -10], crs=ccrs.PlateCarree())
    ax.set_title(f"SPEI-12, lag {lag}, {model}")
    filename = f"/eejit/home/6196306/Data/Figures/Remake_ch3/Lagged_SVD_{reg}_{var}_{region}_lag={lag}_{data2_name}_vt-pattern_mode={mode}_{model}_reverse_v3"
    fig.savefig(f"{filename}.jpg", dpi=1200, bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")

#%% Calculate correlations between time lags
print("r -18 and -12:", xr.corr(svd[-18][3], svd[-12][3]).values)
print("r -18 and -6:", xr.corr(svd[-18][3], svd[-6][3]).values)
print("r -18 and 0:", xr.corr(svd[-18][3], svd[0][3]).values)
print("r -18 and +6:", xr.corr(svd[-18][3], svd[6][3]).values)
print("r -12 and -6:", xr.corr(svd[-12][3], svd[-6][3]).values)
print("r -12 and 0:", xr.corr(svd[-12][3], svd[0][3]).values)
print("r -12 and +6:", xr.corr(svd[-12][3], svd[6][3]).values)
print("r -6 and 0:", xr.corr(svd[-6][3], svd[0][3]).values)
print("r -6 and +6:", xr.corr(svd[-6][3], svd[6][3]).values)
print("r 0 and +6:", xr.corr(svd[0][3], svd[6][3]).values)

