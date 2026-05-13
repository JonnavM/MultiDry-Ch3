#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 15:45:56 2025

@author: Jonna van Mourik

Calculate PET and pr per region and with 12 month running mean
"""
import xarray as xr
import pandas as pd

models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]
dir = "/your/directory/here/"

def force_cftime_to_datetime64(da):
    # Converts cftime.datetime to datetime64[ns] by string cast (lossy but safe)
    da['time'] = xr.DataArray(pd.to_datetime(da['time'].astype(str)), coords=[da['time']], dims="time")
    da['time'] = da.indexes['time'].normalize()
    da = da.resample(time="MS").mean()
    return da

PET_list = []
PR_list = []
for model in models:
    pet_model = xr.open_mfdataset(dir+f"PET/{model}/new_data/detrend/pet_mon-detrend_{model}_historical_r*i1p1f1_1850-2014_1x1.nc", combine="nested", concat_dim="member").pet.expand_dims(model=[model])
    pet_model = force_cftime_to_datetime64((pet_model))
    pr_model = xr.open_mfdataset(dir+f"CMIP6/{model}/detrend/pr_mon-detrend_{model}_historical_r*i1p1f1_1850-2014_1x1.nc", combine="nested", concat_dim="member").pr.expand_dims(model=[model])
    pr_model = force_cftime_to_datetime64(pr_model)
    PET_list.append(pet_model)
    PR_list.append(pr_model)
PET = xr.concat(PET_list, dim="model", coords="minimal")
PR = xr.concat(PR_list, dim="model", coords="minimal")

#Now filter for regions
mask_WEU = xr.open_dataarray(dir+"masks/1x1/mask_WEU_1x1.nc")
mask_SA = xr.open_dataarray(dir+"masks/1x1/mask_SA_1x1.nc")

PET_WEU = PET.where(mask_WEU>=0.9).mean(dim=("lat", "lon"))
PET_SA = PET.where(mask_SA>=0.9).mean(dim=("lat", "lon"))
pr_WEU = PR.where(mask_WEU>=0.9).mean(dim=("lat", "lon"))
pr_SA = PR.where(mask_SA>=0.9).mean(dim=("lat", "lon"))

#also calculate for running mean
PET12_WEU = PET_WEU.rolling(time=12)
PET12_SA = PET_SA.rolling(time=12)
pr12_WEU = pr_WEU.rolling(time=12)
pr12_SA = pr_SA.rolling(time=12)

#%% Save both - run this in queue
def save_to_nc(ts, name):
    print(f"Saving {name}")
    return ts.to_netcdf(dir+f"{name}.nc")

save_to_nc(PET_WEU, "PET-WEU_MM")
save_to_nc(pr_WEU, "PR-WEU_MM")
save_to_nc(PET_SA, "PET-SA_MM")
save_to_nc(pr_SA, "PR-SA_MM")

#%% Load in previous data
PET_WEU = xr.open_dataarray(dir+"PET-WEU_MM.nc")
PR_WEU = xr.open_dataarray(dir+"PR-WEU_MM.nc")
PET_SA = xr.open_dataarray(dir+"PET-SA_MM.nc")
PR_SA = xr.open_dataarray(dir+"PR-SA_MM.nc")

PET12_WEU = PET_WEU.rolling(time=12).mean("time")
PET12_SA = PET_SA.rolling(time=12).mean("time")
PR12_WEU = PR_WEU.rolling(time=12).mean("time")
PR12_SA = PR_SA.rolling(time=12).mean("time")


save_to_nc(PET12_WEU, "PET12-WEU_MM")
save_to_nc(PR12_WEU, "PR12-WEU_MM")
save_to_nc(PET12_SA, "PET12-SA_MM")
save_to_nc(PR12_SA, "PR12-SA_MM")


