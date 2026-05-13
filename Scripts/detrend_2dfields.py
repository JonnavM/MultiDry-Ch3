#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:55:58 2025

@author: Jonna van Mourik
Detrenden vd data
"""
import xarray as xr
import numpy as np
#import matplotlib.pyplot as plt
import sys
import argparse

#%% parse commands
parser = argparse.ArgumentParser(description="Calculate PET using Penman-Monteith")
parser.add_argument("--model", required=True, type=str, help="CMIP6 model name (e.g. CanESM5)")
args = parser.parse_args()

model = args.model

#%%Import data
dir = "/your/directory/here/"

base = [1850, 2014]

def detrend_gmst(da, da_gmst, var):
    """
    Detrend monthly data (da) using annual GMST (da_gmst).
    Supports both 1D (time) and 3D (time, lat, lon) arrays.
    """
    # Stack spatial dims if needed
    spatial_dims = [dim for dim in da.dims if dim != 'time']
    stacked = False
    if spatial_dims:
        da = da.stack(z=spatial_dims)#.T
        stacked = True

    da_mon = da  # assume monthly already
    da_trend_mon = xr.full_like(da_mon, fill_value=np.nan)

    for month in range(1, 13):
        print(month, flush=True)
        # Indices for this calendar month
        idx = da_mon['time'].dt.month == month
        da_thismonth = da_mon.where(idx, drop=True)
        
        # Regression per point
        def fit_trend(y):
            mask = np.isfinite(y) & np.isfinite(da_gmst)
            if mask.sum() < 2:
                return np.full_like(y, np.nan)
            coeffs = np.polyfit(da_gmst, y[mask], 1)
            fit = np.polyval(coeffs, da_gmst)
            return fit

        da_fit = xr.apply_ufunc(
            fit_trend,
            da_thismonth,
            input_core_dims=[["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True}
        )

        # Get base period mask
        years = da_thismonth['time'].dt.year
        base_mask = (years >= base[0]) & (years <= base[1])
        base_mean = da_fit.where(base_mask).mean(dim='time')

        # Store trend
        if var in ['pr', 'pet', "z500", "mrso", "psl"]:
            trend = base_mean / da_fit
        else:
            trend = da_fit - base_mean

        # Insert values into correct positions
        da_trend_mon.loc[dict(time=da_thismonth['time'])] = trend.transpose(*da_thismonth.dims)

    # Unstack if needed
    if stacked:
        da = da.unstack('z')
        da_trend_mon = da_trend_mon.unstack('z')

    # Detrend
    if var in ['pr', 'pet', 'z500', "psl", "mrso"]:
        da_detrend = da * da_trend_mon
        if var == 'pr':
            da_detrend = da_detrend.where(da_detrend > 0, 0)
        method = 'scaled'
    else:
        da_detrend = da - da_trend_mon
        method = 'added'

    return da_detrend, da_trend_mon, method


#%% Parallel

def process_model_member(model, r, var_name):
    print(f"Processing model: {model}, member: r{r}", flush=True)

    if var_name == "pet":
        file_prefix = f"{dir}PET/{model}/"
        var = xr.open_dataset(file_prefix+f"new_data/pm_fao56_r{r}_1850-2014_monthly_{model}_hist_1x1.nc", chunks={"time": 60, "lat":90, "lon":90}).__xarray_dataarray_variable__#.PM_FAO_56
    else:
        file_prefix = f"{dir}CMIP6/{model}/"
        if var_name == "tos":
            var = xr.open_dataset(file_prefix+f"tos_Omon_{model}_historical_r{r}i1p1f1_gn_1850-2014_1x1.nc", chunks={"time": 60, "lat":90, "lon":90}).tos
        elif var_name == "pr":
            var = xr.open_dataset(file_prefix+f"1x1/pr_Amon_{model}_historical_r{r}i1p1f1_gr_1850-2014_1x1.nc").pr
        elif var_name == "tos":
            var = xr.open_dataset(file_prefix+f"1x1/tos_Omon_{model}_historical_r{r}i1p1f1_gr_1850-2014_1x1.nc").tos
        elif var_name == "mrso":
            var = xr.open_dataset(file_prefix++f"1x1/mrso_Lmon_{model}_historical_r{r}i1p1f1_gr_1850-2014_1x1.nc").mrso
        elif var_name == "z500":  
            if model=="CanESM5":
                var = xr.open_dataset(dir+f"CMIP6/{model}/1x1/zg500_AERmon_{model}_historical_r{r}i1p1f1_gn_1850-2014_1x1.nc", chunks={"time": 60, "lat":90, "lon":90}).zg500
            elif model=="CESM2" or model=="MIROC6":
                var = xr.open_dataset(dir+f"CMIP6/{model}/1x1/zg500_AERmon_{model}_historical_r{r}i1p1f1_gn_1850-2014_1x1.nc", chunks={"time": 60, "lat":90, "lon":90}).zg_500
            elif model=="MPI-ESM1-2-LR": 
                var = xr.open_dataset(dir+f"CMIP6/{model}/1x1/zg500_mon_{model}_historical_r{r}i1p1f1_gn_1850-2014_1x1.nc", chunks={"time": 60, "lat":90, "lon":90}).zg500
        elif var_name=="psl":    
            if model=="MIROC6" or model=="MPI-ESM1-2-LR" or model=="ACCESS-ESM1-5":
                var = xr.open_dataset(file_prefix + f"1x1/psl_mon_{model}_historical_r{r}i1p1f1_gn_1850-2014_1x1.nc", chunks={"time": 60}).psl
            else:
                var = xr.open_dataset(file_prefix + f"1x1/psl_Amon_{model}_historical_r{r}i1p1f1_gn_1850-2014_1x1.nc", chunks={"time": 60}).psl
    da_gmst = xr.open_dataarray(dir + f"CMIP6/{model}/global/tas_yr_{model}_historical_r{r}i1p1f1_global.nc", chunks={"time": -1})
    da_gmst = da_gmst.sel(time=slice("1850", "2014"))

    # Smooth GMST and fit polynomial
    da_gmst_fit_para = np.polyfit(da_gmst.time.dt.year, da_gmst, 5)
    da_gmst_fit = np.polyval(da_gmst_fit_para, da_gmst.time.dt.year)
    da_gmst_fit = xr.DataArray(da_gmst_fit, coords=da_gmst.coords)
    print("GMST smoothed and fitted", flush=True)
    # Select relevant data and detrend
    da = var.sel(time=slice("1850", "2014"))
    da_detrend, da_trend, method = detrend_gmst(da, da_gmst_fit, var=var_name)
    print("detrend function applied", flush=True)
    # Set attributes
    da_detrend.name = var_name
    da_detrend.attrs = da.attrs
    da_detrend.attrs['detrended'] = f"Removed trend, base-period {base[0]}-{base[1]}"
    da_trend = da_trend.rename({'time': 'time_mon'})
    da_trend.name = 'trend'
    if method == 'added':
        if 'units' in da.attrs:
            da_trend.attrs['units'] = da.units
    else:
        da_trend.attrs['units'] = '-'
    da_detrend.attrs['trend'] = f"Linear fit with GMST, per month, {method}"
    da_detrend.attrs['base-period'] = f"{base[0]}-{base[1]}"

    # Save output
    detrend_path = file_prefix+f"new_data/detrend/{var_name}_mon-detrend_{model}_historical_r{r}i1p1f1_1850-2014_1x1.nc"
    trend_path = file_prefix+f"new_data/trend/{var_name}_mon-trend_{model}_historical_r{r}i1p1f1_1850-2014_1x1.nc"
    da_detrend.to_netcdf(detrend_path)
    print("Save detrend", flush=True)
    da_trend.to_netcdf(trend_path)
    print("Save trend", flush=True)

    #return f"Finished: {model} r{r}"
    return model, r, da_detrend, da_trend, var_name

members = [1,2,3,4,5,6,7,8,9,10]
var_name = "pet"  
tasks = [process_model_member(model, r, var_name) for r in members]

