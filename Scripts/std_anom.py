#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 13:47:05 2025

@author: Jonna van Mourik
Standardize data
"""
import xarray as xr
import dask

def std_anomalies(ds):
    "Function to standardize a dataset with respect to its month, the standard deviations, and the gridcells"
    ds_clim = ds.groupby('time.month').mean(dim='time')
    ds_anom = ds.groupby('time.month') - ds_clim
    ds_std = ds_anom.groupby('time.month').std("time")
    ds_std_matched = ds_std.sel(month=ds_anom["month"])
    ds_anom_norm = (ds_anom/ds_std_matched)*area_factor
    return ds_anom_norm    

models=["ACCESS-ESM1-5", "CanESM5", "CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]
dir = "/your/directory/here/"
#Load in area per cell
areacello = xr.open_dataarray(dir+"CMIP6/areacello.nc")
#Calculate the area factor 
area_factor = areacello/areacello.max()

var_name = "pet"
for model in models:
    print(model, flush=True)
    if var_name=="pet":
        var = xr.open_mfdataset(dir+f"PET/{model}/new_data/detrend/pet_mon-detrend_{model}_historical_r*i1p1f1_1850-2014_1x1.nc", combine="nested", concat_dim="member").pet.resample(time="1MS").mean()
    else:
        var = xr.open_mfdataset(dir+f"CMIP6/{model}/detrend/{var_name}_mon-detrend_{model}_historical_r*i1p1f1_1850-2014_1x1.nc", combine='nested', concat_dim='member').psl.resample(time="1MS").mean()
    var_stdanom = std_anomalies(var)
    print("Save to NetCDF", flush=True)
    if var_name=="pet":
        var_stdanom.to_netcdf(dir+f"PET/{model}/new_data/detrend/std_anom_{var_name}_mon-detrend_{model}_historical_1850-2014_1x1.nc")
    else:
        path = dir+f"CMIP6/{model}/detrend/std_anom_{var_name}_mon-detrend_{model}_historical_1850-2014_1x1.nc"
        #delayed = var_stdanom.to_netcdf(path, compute=False)
        #dask.compute(delaye)
        var_stdanom.to_netcdf(path)
    print("Done", flush=True)