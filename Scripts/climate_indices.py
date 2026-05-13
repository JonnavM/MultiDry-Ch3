#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:35:44 2024

@author: Jonna van Mourik
Calculate climate indices
"""
#Import things
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

#Define model and load in data -> Moet eerst nog geregrid worden
model = "MPI-ESM1-2-LR" # Repeat per model
dir = "/your/directory/here/"
SST_ds = xr.open_mfdataset(dir+"tos_Omon_MPI-ESM1-2-LR_historical_r*i1p1f1_gn_1850-2014_1x1.nc", combine='nested', concat_dim='member').tos
psl_ds = xr.open_mfdataset(dir+"psl_mon_MPI-ESM1-2-LR_historical_r*i1p1f1_gn_1850-2014_1x1.nc", combine='nested', concat_dim='member').psl

areacello = xr.open_dataarray("/data/droughtTeam/CMIP6/areacello.nc")
area_factor = areacello/areacello.max()

def normalize(ds):
    "Function to normalize a dataset with respect to its month, the standard deviations, and the gridcells"
    ds_clim = ds.groupby('time.month').mean(dim='time')
    ds_anom = ds.groupby('time.month') - ds_clim
    ds_std = ds_anom.groupby('time.month').std("time")
    ds_std_matched = ds_std.sel(month=ds_anom["month"])
    ds_anom_norm = (ds_anom/ds_std_matched)*area_factor
    return ds_anom_norm 
SST_norm = normalize(SST_ds)
psl_norm = normalize(psl_ds)

#%%Select masks for regions

#Compute anomalies
def nino_index(SST, version):
    if version==12:
        SST_nino = SST.sel(lat=slice(-10, 0), lon=slice(-90, -80))
    elif version==3:
        SST_nino = SST.sel(lat=slice(-5, 5), lon=slice(-150, -90))
    elif version==34:
        SST_nino = SST.sel(lat=slice(-5, 5), lon=slice(-170, -120))
    elif version==4:
        SST_nino = SST.sel(lat=slice(-5, 5)).where((SST.lon>=160) | (SST.lon<=-150), drop=True)
    gb = SST_nino.groupby("time.month")
    SST_nino_anom = gb-gb.mean(dim="time")
    index_weighted = SST_nino_anom.weighted(areacello).mean(dim=("lat", "lon"))
    #Apply a 5 month smoothing
    index_rolling = index_weighted.rolling(time=5, center=True).mean()
    #Calculate std
    std_dev_nino = SST_nino.std()
    #Normalize the data
    nino_index = index_rolling / std_dev_nino
    print("Nino version "+str(version)+" calculated")
    return nino_index

nino_12 = nino_index(SST_ds, 12)
nino_3 = nino_index(SST_ds, 3)
nino_34 = nino_index(SST_ds, 34)
nino_4 = nino_index(SST_ds, 4)

#Save the timeseries
def save_to_nc(ts, name):
    return ts.to_netcdf(f"/data/droughtTeam/CMIP6/{model}/climate_indices/{name}_mon_{model}_historical_1850-2014.nc")

save_to_nc(nino_4, "nino_4")

#%% Atlantic section Benguela Nino index, based on Bachelery et al. (2025)
ATL3_box = dict(lat=slice(-3,3), lon=slice(-20,0))
CABA_box = dict(lat=slice(-20,-10), lon=slice(2,4))
MEQUE_box = dict(lat=slice(-28,-10), lon=slice(8,15))

def Atlantic_nino_index(SST, box):
    #Apply 1-2 decentered weighted running average
    weights = np.array([0.25, 0.5, 0.25])
    SST_weighted = SST.rolling(time=3, center=False).construct("window").dot(xr.DataArray(weights, dims="window"))
    return SST_weighted.sel(**box).mean(dim=["lat", "lon"])
    
ANi = Atlantic_nino_index(SST_norm, ATL3_box)
BNi = Atlantic_nino_index(SST_norm, CABA_box)
Meque_BNi = Atlantic_nino_index(SST_norm, MEQUE_box)

save_to_nc(ANi, "Atlantic-nino-norm")
save_to_nc(BNi, "Benguela-nino-norm")
save_to_nc(Meque_BNi, "Benguela-nino-Meque-norm")

#%% Indian ocean dipole
def compute_IOD(sst):
    west = sst.sel(lat=slice(-10, 10), lon=slice(50, 70)).mean(dim=("lat", "lon"))
    east = sst.sel(lat=slice(-10, 0), lon=slice(90, 110)).mean(dim=("lat", "lon"))
    return west - east

IOD = compute_IOD(SST_norm)
save_to_nc(IOD, "IOD-norm")
#%% Subtropical Indian Ocean dipole
def compute_SIOD(sst):
    sw = sst.sel(lat=slice(-37, -27), lon=slice(55, 65)).mean(dim=("lat", "lon"))
    se = sst.sel(lat=slice(-28, -18), lon=slice(90, 100)).mean(dim=("lat", "lon"))
    return sw - se

SIOD = compute_SIOD(SST_norm)
save_to_nc(SIOD, "SIOD-norm")
#%% Sea level Pressure
#SAM index by Gong and Wang
#SAM = P*40s-P*65s where P* indicates the normalized zonal-mean SLP at a particular latitude (40S and 65S most strongly anticorrelate). 
#Normalize per season (in paper only DJF is used)
def compute_SAM(psl):
    psl_40S = psl.sel(lat=-40, method="nearest").mean(dim="lon")
    psl_65S = psl.sel(lat=-65, method="nearest").mean(dim="lon")
    return psl_40S - psl_65S
SAM = compute_SAM(psl_norm)
save_to_nc(SAM, "SAM-norm")

#%% Botswana High pressure centre
def compute_BH(psl):
    bh = psl.sel(lat=slice(-25, -20), lon=slice(20, 30)).mean(dim=("lat", "lon"))
    return bh

BH = compute_BH(psl_norm)
save_to_nc(BH, "Botswana-high-norm")

#%% South Atlantic Subtropical Dipole
def compute_SASD(sst):
    north = sst.sel(lat=slice(-25, -15), lon=slice(-20, 0)).mean(dim=("lat", "lon"))
    south = sst.sel(lat=slice(-40, -30), lon=slice(-30, -10)).mean(dim=("lat", "lon"))
    return north - south

SASD = compute_SASD(SST_norm)
save_to_nc(SASD, "SASD-norm")

#%% South western Indian Ocean index
def compute_SWIO(sst):
    swio = sst.sel(lat=slice(-15, -5), lon=slice(55, 75)).mean(dim=("lat", "lon"))
    return swio

SWIO = compute_SWIO(SST_norm)
save_to_nc(SWIO, "SWIO-norm")

#%% Ningaloo nino index
def compute_NNI(sst):
    nni = sst.sel(lat=slice(-28, -22), lon=slice(108, 116))
    return nni

NNI = compute_NNI(SST_norm)
save_to_nc(NNI, "NNI-norm")