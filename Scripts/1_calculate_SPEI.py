#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:37:12 2024

@author: Jonna van Mourik
"""

import xclim
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from xclim import indices
from xclim.core import units
from xclim.indices import standardized_precipitation_evapotranspiration_index
import spei as si  # si for standardized index
import pandas as pd
import cftime

xr.set_options(keep_attrs=True)

#%% Define SPEI
dir_path = "/your/directory/here/"
def SPEI_region(region_name, prec, pet, spei_period, offset, cal_start, cal_end, dir):
    if region_name == "IND": #IND, AUS, WEU, SA, ARG, CAL
        reg_lat = slice(10,47)
        reg_lon = slice(60,110)
        region_mask = xr.open_dataset(dir_path+"masks/1x1/mask_IND_1x1.nc").Band1#.interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "AUS":
        reg_lat = slice(-50, -10)
        reg_lon = slice(120, 165)
        region_mask = xr.open_dataset(dir_path+"masks/1x1/mask_AUS_1x1.nc").Band1#.interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "WEU":
        reg_lat = slice(20, 70)
        reg_lon = slice(-30,30)
        region_mask = xr.open_dataset(dir_path+"masks/1x1/mask_WEU_1x1.nc").Band1#.interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "SA":
        reg_lat = slice(-45,-12)
        reg_lon = slice(2, 50)
        region_mask = xr.open_dataset(dir_path+"masks/1x1/mask_SA_1x1.nc").Band1#.interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "ARG":
        reg_lat = slice(-45, -25)
        reg_lon = slice(285, 305)
        region_mask = xr.open_dataset(dir_path+"masks/1x1/mask_ARG_1x1.nc").Band1#.interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)
    elif region_name == "CAL":
        reg_lat = slice(15, 60)
        reg_lon = slice(222, 260)
        region_mask = xr.open_dataset(dir_path+"masks/1x1/mask_CAL_1x1.nc").Band1#.interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest').sel(lat=reg_lat, lon=reg_lon)
        
        
    prec_region = prec.where(region_mask>=0.9)
    pet_region = pet.where(region_mask>0.9)
    
    prec_region_mean = prec_region.mean(dim = ["lon","lat"])
    pet_region_mean = pet_region.mean(dim = ["lon","lat"])
    if pet_region_mean.time.dtype == object:
        pet_region_mean["time"]  = pet_region_mean.indexes["time"].to_datetimeindex()

    
    pe_region_mean = prec_region_mean.assign_attrs(units='mm/d') - pet_region_mean.assign_attrs(units='mm/d')
    print("calculating spei")
    SPEI = standardized_precipitation_evapotranspiration_index(pe_region_mean, window = spei_period, dist = "fisk",freq= "MS", offset=offset,  cal_start = cal_start, cal_end = cal_end)

    del SPEI.attrs['freq']
    del SPEI.attrs['time_indexer']
    del SPEI.attrs['units']
    del SPEI.attrs['offset']
    print("saving")
    SPEI.to_netcdf(path = dir_path+f"SPEI/{model}/new_data/" + dir)
    print("done")
    
    SPEI12 = xr.open_dataset(dir_path+f"SPEI/{model}/new_data/" + dir).__xarray_dataarray_variable__
    df_spei = SPEI12.to_pandas()
    
    f, ax = plt.subplots(1, 1, figsize=(16, 9), sharex=False)
    si.plot.si(df_spei[11:], ax=ax)
    [ax.set_ylabel(n, fontsize=14) for i, n in enumerate(["SPEI"])]
    
    df_pre = prec_region_mean.to_pandas()
    df_pet = pet_region_mean.to_pandas()
    df_pe = pe_region_mean.to_pandas()
    
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    df_pre[:-2].plot(ax=ax[0], legend=True, grid=True, label = "Total Precipitation (mm)").legend(loc='upper left')
    df_pet.plot(ax=ax[1], color="C1", legend=True, grid=True, label = "Potential Evapotranspiration (mm/day)")
    df_pe.plot(ax=ax[2], color="k", legend=True, grid=True, label = "PREC - PET")
    
def SPEI_global(prec, pet, spei_period, offset, cal_start, cal_end, dir):
    pe = prec.assign_attrs(units='mm/d') - pet.assign_attrs(units='mm/d')
    print("Calculating SPEI")
    SPEI = standardized_precipitation_evapotranspiration_index(pe, window = spei_period, dist = "fisk",freq= "MS", offset=offset,  cal_start = cal_start, cal_end = cal_end)
    del SPEI.attrs['freq']
    del SPEI.attrs['time_indexer']
    del SPEI.attrs['units']
    del SPEI.attrs['offset']
    print("saving")
    SPEI.to_netcdf(path = dir_path+f"SPEI/{model}/new_data/" + dir)
    print("done")
    
def convert_time(da):
    """Convert time coordinate to datetime64[ns] if stored as cftime.datetime."""
    if isinstance(da.time.values[0], cftime.datetime):
        da = da.assign_coords(time=pd.to_datetime(da.time.values.astype(str)))  # Convert to string first
    return da
#%% Load in data
models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]
# landmask, since we don't need data over the oceans
landmask = xr.open_dataset(dir_path+"masks/1x1/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_1x1.nc").sftlf

#%% Calculate per member
year = "1850"
for model in models:
    print(model, flush=True)
    if model=="EC-Earth3":
        members = [1, 2, 4, 5, 6, 7, 9, 10, 11, 13]
        grid="gr"
    else: 
        members = np.arange(1,11)
        grid="gn"
    for member in members:
        print(member, flush=True)
        # Precipitation, is in mm/s, but we need it in mm/day.
        total_prec_mm = (xr.open_dataset(dir_path+f"CMIP6/{model}/1x1/pr_Amon_{model}_historical_r{member}i1p1f1_{grid}_1850-2014_1x1con.nc").pr*3600*24).resample(time="MS").mean()
        landmask_interpolated = landmask.interp(lat=total_prec_mm.lat, lon=total_prec_mm.lon, method='nearest')
        total_prec_mm = convert_time(total_prec_mm.where(landmask_interpolated>=50))
        # PET, resample to monthly values
        pet = xr.open_dataset(dir_path+f"PET/{model}/new_data/pm_fao56_r{member}_1850-2014_monthly_{model}_hist_1x1.nc").__xarray_dataarray_variable__.where(landmask_interpolated>=50).resample(time="MS").mean()
        
        # Calculate SPEI
        # Set the offset, specify start- and end of the calibration period. Mean is zero between these dates
        prec = total_prec_mm
        pet = convert_time(pet)
        spei_period = 12
        offset = '20 mm/d'
        cal_start = f"{year}-01-01"
        cal_end = "2014-12-31"
        #Per region
        region_name = ["CAL", "WEU", "IND", "ARG", "SA", "AUS"]
        dir = [f"SPEI12_monthly_{year}_2014_r"+str(member)+"_"+str(model)+"_CAL.nc", f"SPEI12_monthly_{year}_2014_r"+str(member)+"_"+str(model)+"_WEU.nc",
               f"SPEI12_monthly_{year}_2014_r"+str(member)+"_"+str(model)+"_IND.nc", f"SPEI12_monthly_{year}_2014_r"+str(member)+"_"+str(model)+"_ARG.nc", 
               f"SPEI12_monthly_{year}_2014_r"+str(member)+"_"+str(model)+"_SA.nc", f"SPEI12_monthly_{year}_2014_r"+str(member)+"_"+str(model)+"_AUS.nc"]
        #region_name = ["WEU", "SA"]
        #dir = ["SPEI12_monthly_detrend_1850_2014_r"+str(member)+"_"+str(model)+"_WEU.nc", "SPEI12_monthly_detrend_1850_2014_r"+str(member)+"_"+str(model)+"_SA.nc"]
        for i in range(len(region_name)):
            print(region_name[i])
            SPEI_region(region_name[i], prec, pet, spei_period, offset, cal_start, cal_end, dir[i])
            
        #Global
        dir = f"SPEI12_monthly_{year}_2014_r"+str(member)+"_"+str(model)+"_1x1.nc"
        
        SPEI_global(prec, pet, spei_period, offset, cal_start, cal_end, dir)
