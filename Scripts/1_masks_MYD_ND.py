#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:13:29 2024

@author: Jonna van Mourik
Calculate MYD and ND periods
"""
import xarray as xr
import sys
sys.path.append("/your/directory/here/") 
from functions import MYD, ND, mask_MYD, mask_ND
import numpy as np

#Data
dir = "/your/directory/here/"
models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]

for model in models:
    print(model, flush=True)
    SPEI_WEU = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/new_data/detrend/SPEI12_monthly_detrend_1850_2014_r*_"+str(model)+"_WEU.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__#.sel(time=slice("1950", "2014"))
    SPEI_SA = xr.open_mfdataset(dir+"SPEI/"+str(model)+"/new_data/detrend/SPEI12_monthly_detrend_1850_2014_r*_"+str(model)+"_SA.nc", combine='nested', concat_dim='member').__xarray_dataarray_variable__#.sel(time=slice("1950", "2014"))
    
    #Masks
    mask_WEU = xr.open_dataset(dir+"masks/1x1/mask_WEU_1x1.nc").Band1
    mask_SA = xr.open_dataset(dir+"masks/1x1/mask_SA_1x1.nc").Band1
    
    #South Africa
    mask_MYD_SA = mask_MYD(SPEI_SA, "SA", oneyear=False)
    mask_ND_SA = mask_ND(SPEI_SA, "SA", oneyear=False)
    
    #Western Europe
    mask_MYD_WEU = mask_MYD(SPEI_WEU, "WEU", oneyear=False)
    mask_ND_WEU = mask_ND(SPEI_WEU, "WEU", oneyear=False)
    
    #South Africa
    mask_MYD_SA_1yr = mask_MYD(SPEI_SA, "SA", oneyear=True)
    mask_ND_SA_1yr = mask_ND(SPEI_SA, "SA", oneyear=True)
    
    #Western Europe
    mask_MYD_WEU_1yr = mask_MYD(SPEI_WEU, "WEU", oneyear=True)
    mask_ND_WEU_1yr = mask_ND(SPEI_WEU, "WEU", oneyear=True)
    
    # Save masks
    def saveList(myList,filename):
    # the filename should mention the extension 'npy'
        np.save(filename,myList)
    print("Saved successfully!", flush=True)
    
    reg_name = ["WEU", "SA"]    
    reg_mask_MYD = [mask_MYD_WEU, mask_MYD_SA]
    reg_mask_MYD_1yr = [mask_MYD_WEU_1yr, mask_MYD_SA_1yr]
    reg_mask_ND = [mask_ND_WEU, mask_ND_SA]
    reg_mask_ND_1yr = [mask_ND_WEU_1yr, mask_ND_SA_1yr]
    for i in range(len(reg_name)):        
        reg_mask_MYD[i].to_netcdf(dir+"masks/"+str(model)+"/new_data/detrend/mask_MYD_"+str(model)+"_"+str(reg_name[i])+"_1850-2014.nc")#"_1yprior.nc")
        reg_mask_ND[i].to_netcdf(dir+"masks/"+str(model)+"/new_data/detrend/mask_ND_"+str(model)+"_"+str(reg_name[i])+"_1850-2014.nc")#"_1yprior.nc")
        reg_mask_MYD_1yr[i].to_netcdf(dir+"masks/"+str(model)+"/new_data/detrend/mask_MYD_"+str(model)+"_"+str(reg_name[i])+"_1850-2014_1yprior.nc")
        reg_mask_ND_1yr[i].to_netcdf(dir+"masks/"+str(model)+"/new_data/detrend/mask_ND_"+str(model)+"_"+str(reg_name[i])+"_1850-2014_1yprior.nc")
