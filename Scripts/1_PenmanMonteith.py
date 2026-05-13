#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:01:38 2024

PET for any dataset
"""
import numpy as np
import pyet as pyet
import xarray as xr

#%% Specify years, members, model
model = "MPI-ESM1-2-LR"
members = np.arange(1,11,1)

if model == "MPI-ESM1-2-LR":
    years_start = np.arange(1850, 2015, 20)
    years_end = np.arange(1869, 2014, 20)
    years_end = np.append(years_end, 2014)
elif model == "EC-Earth3": 
    years_start = np.arange(1850, 2015, 1)
    years_end = years_start 
elif model == "ACCESS-ESM1-5":
    years_start = np.arange(1850, 2015, 50)
    years_end = np.arange(1899, 2014, 50)
    years_end = np.append(years_end, 2014)
elif model == "CanESM5":
    years_start = [1850]
    years_end = [2014]
elif model == "MIROC6":
    years_start = np.arange(1850, 2015, 10)
    years_end = np.arange(1899, 2014, 10)
    years_end = np.append(years_end, 2014)
    
#%% Calculate PET
for member in members:
    for i in range(len(years_start)):   
        print(member, years_start[i])        
        #Load in datasets    
        dir = "/your/directory/here/CMIP/"+str(model)+"/"
        tmax = xr.open_dataset(dir+"tasmax_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").tasmax-273.15 # Daily maximum temperature [°C]    
        tmin = xr.open_dataset(dir+"tasmin_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").tasmin-273.15 # Daily minimum temperature [°C]
        tmean = xr.open_dataset(dir+"tas_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").tas-273.15 # Daily mean temperature [°C]
        print("Temperature loaded in")
        
        rh_mean = xr.open_dataset(dir+"hurs_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").hurs # Daily mean relative humidity [%]
        rh_max = xr.open_dataset(dir+"hursmax_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").hursmax # Daily max relative humidity [%]
        rh_min = xr.open_dataset(dir+"hursmin_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").hursmin # Daily min relative humidity [%]
        print("Relative humidity calculated")
        
        uz = xr.open_dataset(dir+"sfcWind_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").sfcWind  # Wind speed at 10 m [m/s]
        z = 10  # Height of wind measurement [m]
        wind_fao56 = uz * 4.87 / np.log(67.8*z-5.42)  # wind speed at 2 m after Allen et al., 1998
        print("Wind loaded in and calculated for 2m instead of 10m")
        
        p = xr.open_dataset(dir+"psl_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").psl*1e-3 #Surface pressure [kPa]
        print("Surface pressure loaded in")
        
        drs = xr.open_dataset(dir+"rsds_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").rsds*3600*24*1e-6 # Compute solar radiation [MJ/m2day]
        drt = xr.open_dataset(dir+"rlds_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").rlds*3600*24*1e-6 #thermal radiation [MJ/m2day]

        urs = xr.open_dataset(dir+"rsus_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").rsus*3600*24*1e-6 # Compute solar radiation [MJ/m2day]
        urt = xr.open_dataset(dir+"rlus_day_"+str(model)+"_historical_r"+str(member)+"i1p1f1_gn_"+str(years_start[i])+"0101-"+str(years_end[i])+"1231.nc").rlus*3600*24*1e-6 #thermal radiation [MJ/m2day]
    
        nrs = drs - urs
        nrt = drt - urt
        rn = nrs + nrt #Turn into + if rt<0! Net radiation
        time = tmean.time
        lat = tmean.lat
        elevation = 2
        print("Radiation loaded in and calculated")
        
        pm_fao56 = pyet.pm_fao56(tmean, wind=wind_fao56, rs=nrs, rn=rn, pressure=p, elevation=elevation, lat=lat, tmax=tmax, tmin=tmin, rh=rh_mean, rhmax=rh_max, rhmin=rh_min)
        print("Penman-Monteith calculated")
    
        pm_fao56.to_netcdf("/CMIP/PET/MPI/pm_fao56_r"+str(member)+"_"+str(years_start[i])+"-"+str(years_end[i])+"daily_"+str(model)+"_hist.nc")
        print("Penman-Monteith saved to netCDF")
