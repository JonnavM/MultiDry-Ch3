#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:35:44 2024

@author: Jonna van Mourik
Calculate climate indices
"""
#Import things
import xarray as xr
import numpy as np
import pandas as pd
from eofs.xarray import Eof
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
"""
#Define model and load in data -> Moet eerst nog geregrid worden
model = "ACCESS-ESM1-5"
dir = "/your/directory/here/"
SST_ds = xr.open_mfdataset(dir+f"CMIP6/{model}/detrend/tos_mon-detrend_{model}_historical_r*i1p1f1_1850-2014_1x1.nc", combine='nested', concat_dim='member').tos
psl_ds = xr.open_mfdataset(dir+f"CMIP6/{model}/detrend/psl_mon-detrend_{model}_historical_r*i1p1f1_1850-2014_1x1.nc", combine='nested', concat_dim='member').psl

areacello = xr.open_dataarray(dir+"CMIP6/areacello.nc")
area_factor = areacello/areacello.max()
"""
def force_cftime_to_datetime64(da):
    # Converts cftime.datetime to datetime64[ns] by string cast (lossy but safe)
    da['time'] = xr.DataArray(pd.to_datetime(da['time'].astype(str), format='mixed', errors='coerce'), coords=[da['time']], dims="time")
    da['time'] = da.indexes['time'].normalize()
    da = da.resample(time="MS").mean()
    return da

#Save the timeseries
def save_to_nc(ts, name):
    ts_new = force_cftime_to_datetime64(ts)
    return ts_new.assign_coords(model=model).to_netcdf(f"/Climate_indices/{name}_mon_{model}_historical.nc")

#%% All climate indices

#Compute anomalies
def nino_index(SST, version):
    "https://climatedataguide.ucar.edu/climate-data/overview-climate-indices"
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

# Atlantic section Benguela Nino index, based on Bachelery et al. (2025)
def Atlantic_nino_index(SST, box):
    #Apply 1-2 decentered weighted running average
    weights = np.array([0.25, 0.5, 0.25])
    SST_weighted = SST.rolling(time=3, center=False).construct("window").dot(xr.DataArray(weights, dims="window"))
    return SST_weighted.sel(**box).weighted(areacello).mean(dim=["lat", "lon"])

# Indian ocean dipole, based on Saji et al. (2003)
def compute_IOD(sst):
    west = sst.sel(lat=slice(-10, 10), lon=slice(50, 70)).weighted(areacello).mean(dim=("lat", "lon"))
    east = sst.sel(lat=slice(-10, 0), lon=slice(90, 110)).weighted(areacello).mean(dim=("lat", "lon"))
    return west - east

# Subtropical Indian Ocean dipole, based on Behera et al. (2001)
def compute_SIOD(sst):
    sw = sst.sel(lat=slice(-37, -27), lon=slice(55, 65)).weighted(areacello).mean(dim=("lat", "lon"))
    se = sst.sel(lat=slice(-28, -18), lon=slice(90, 100)).weighted(areacello).mean(dim=("lat", "lon"))
    return sw - se

# South Atlantic Subtropical Dipole, based on Morioka et al. (2011)
def compute_SASD(sst):
    north = sst.sel(lat=slice(-25, -15), lon=slice(-20, 0)).weighted(areacello).mean(dim=("lat", "lon"))
    south = sst.sel(lat=slice(-40, -30), lon=slice(-30, -10)).weighted(areacello).mean(dim=("lat", "lon"))
    return north - south

# South western Indian Ocean index, based on a combination of Huang et al. (2024) and Annamalai et al. (2005)
def compute_SWIO(sst):
    swio = sst.sel(lat=slice(-15, -5), lon=slice(55, 75)).weighted(areacello).mean(dim=("lat", "lon"))
    return swio

# Ningaloo nino index, based on Doi et al. (2013)
def compute_NNI(sst):
    nni = sst.sel(lat=slice(-28, -22), lon=slice(108, 116)).weighted(areacello).mean(dim=("lat", "lon"))
    return nni

# Tripole Pattern Index (TPI) / North Atlantic SST tripole
# Sutton & Hodson (2003):
def compute_TPI(sst):
    mid_NA = sst.sel(lat=slice(20, 40), lon=slice(-80, -20)).weighted(areacello).mean(dim=("lat", "lon")) 
    subpolar = sst.sel(lat=slice(50, 70), lon=slice(-60, -20)).weighted(areacello).mean(dim=("lat", "lon"))
    tropics = sst.sel(lat=slice(0, 20), lon=slice(-60, -20)).weighted(areacello).mean(dim=("lat", "lon"))
    return mid_NA - (subpolar+tropics)/2

# Sea level Pressure
#SAM index by Gong and Wang
#SAM = P*40s-P*65s where P* indicates the normalized zonal-mean SLP at a particular latitude (40S and 65S most strongly anticorrelate). 
#Normalize per season (in paper only DJF is used)
def compute_SAM(psl):
    psl_40S = psl.sel(lat=-40, method="nearest").weighted(areacello.sel(lat=-40, method="nearest")).mean(dim="lon")
    psl_65S = psl.sel(lat=-65, method="nearest").weighted(areacello.sel(lat=-40, method="nearest")).mean(dim="lon")
    return psl_40S - psl_65S


# Botswana High pressure centre, based on Maoyi et al. (2024)
def compute_BH(psl):
    bh = psl.sel(lat=slice(-25, -20), lon=slice(15, 22)).weighted(areacello).mean(dim=("lat", "lon"))
    return bh


# North Atlantic Oscillation
# Jianping, L., Wang, J.X.L. A new North Atlantic Oscillation index and its variability. Adv. Atmos. Sci. 20, 661–676 (2003). https://doi.org/10.1007/BF02915394
def compute_NAO(psl):
    north = psl.sel(lat=65.5, lon=slice(-80, 30)).weighted(areacello.sel(lat=65.5, lon=slice(-80, 30))).mean(dim=("lon"))
    south = psl.sel(lat=35.5, lon=slice(-80, 30)).weighted(areacello.sel(lat=65.5, lon=slice(-80, 30))).mean(dim=("lon"))
    return north-south

# Southern Oscillation index
# https://www.ncei.noaa.gov/access/monitoring/enso/soi#calculation-of-soi
def compute_SOI(psl):
    SLP_T = psl.sel(lat=-17.40, lon=-149.25, method="nearest")
    SLP_D = psl.sel(lat=-12.26, lon=130.50, method="nearest")
    N = len(SLP_T["time"])
    sigma_T = np.sqrt(np.sum(SLP_T-SLP_T.mean("time"))**2/N)
    sigma_D = np.sqrt(np.sum(SLP_D-SLP_D.mean("time"))**2/N)
    sSLP_T = (SLP_T - SLP_T.mean("time"))/sigma_T
    sSLP_D = (SLP_D - SLP_D.mean("time"))/sigma_D
    sigma_m = np.sqrt(np.sum(sSLP_T - sSLP_D)**2/N)
    SOI = (sSLP_T - sSLP_D)/sigma_m
    return SOI

# Greenland pattern, based on Hanna et al. (2016)
def compute_GBI(psl):
    """Compute Greenland Blocking Index (GBI)."""
    region = psl.sel(lat=slice(60, 80), lon=slice(-80, -20))
    
    clim = region.groupby("time.month").mean("time")
    anom = region.groupby("time.month") - clim
    
    return anom.weighted(areacello).mean(("lat", "lon")).rename("GBI")

# Scandinavian blocking
def compute_EOFs(psl, areacello):
    """
    Compute the first 3 EOFs for the North Atlantic sector
    https://climatedataguide.ucar.edu/climate-data/hurrell-north-atlantic-oscillation-nao-index-pc-based
    """
    # Select domain: Eurasia/North Atlantic sector
    region = psl.sel(lat=slice(20, 90), lon=slice(-90, 40))
    area_sub = areacello.sel(lat=slice(20, 90), lon=slice(-90, 40))
    
    # Remove monthly climatology per member
    clim = region.groupby("time.month").mean("time")
    anom = region.groupby("time.month") - clim
    
    # Compute weights
    weights = np.sqrt(area_sub / area_sub.max())

    EOF1_members = []
    EOF2_members = []
    EOF3_members = []

    for m in anom.member:
        print(m)
        da_m = anom.sel(member=m).load()
        da_m = da_m.interpolate_na(dim="time", method="nearest", fill_value="extrapolate")

        # Weighted mean removal (ensures anomalies have zero mean)
        da_m = da_m - da_m.weighted(area_sub).mean(("lat", "lon"))
        # EOF solver
        solver = Eof(da_m, weights=weights)
        # Take EOF2 (EOF1 ~ NAO, EOF2 ~ SCA)
        pc1 = solver.pcs(npcs=3, pcscaling=1)[:, 0]
        pc2 = solver.pcs(npcs=3, pcscaling=1)[:, 1]
        pc3 = solver.pcs(npcs=3, pcscaling=1)[:, 2]
#        eof = solver.eofsAsCorrelation(neofs=3)
        EOF1_members.append(xr.DataArray(pc1, coords={"time": da_m.time, "member": m}, name="EOF1"))
        EOF2_members.append(xr.DataArray(pc2, coords={"time": da_m.time, "member": m}, name="EOF2"))
        EOF3_members.append(xr.DataArray(pc3, coords={"time": da_m.time, "member": m}, name="EOF3"))

    EOF1_all = xr.concat(EOF1_members, dim="member")
    EOF2_all = xr.concat(EOF2_members, dim="member")
    EOF3_all = xr.concat(EOF3_members, dim="member")

    # Standardize
    EOF1_all = (EOF1_all - EOF1_all.mean("time")) / EOF1_all.std("time")
    EOF2_all = (EOF2_all - EOF2_all.mean("time")) / EOF2_all.std("time")
    EOF3_all = (EOF3_all - EOF3_all.mean("time")) / EOF3_all.std("time")

    return EOF1_all, EOF2_all, EOF3_all

# Wave 5/7 on NH ?
def compute_wave_K(z500, lat_band=slice(30,65), k=5): 
    """Compute amplitude of zonal wavenumber k at a given latitude."""
    da_lat = z500.sel(lat=lat_band).weighted(areacello.sel(lat=lat_band)).mean("lat")
    da_anom = da_lat - da_lat.mean("lon")
    
    fft = np.fft.fft(da_anom, axis=da_anom.get_axis_num("lon"))
    amp = np.abs(fft[:, k]) / len(da_lat.lon)
    
    return xr.DataArray(amp, coords={"time": da_lat.time}, name=f"Wave{k}_amplitude")

# Pacific decadal oscillation
def compute_PDO(SST_ds, areacello):
    """
    Compute Pacific Decadal Oscillation (PDO) index for each ensemble member
    from SST (tos) anomalies using EOF analysis.
    Expects: SST_ds(member, time, lat, lon), areacello(lat, lon)
    """
    #Change longitudes to 0-360
    SST_ds = SST_ds.assign_coords(lon=((SST_ds.lon + 360) % 360)).sortby("lon")
    areacello = areacello.assign_coords(lon=((areacello.lon + 360) % 360)).sortby("lon")
    # Subset to North Pacific (110°E–100°W, 20–70°N)
    region = SST_ds.sel(lat=slice(20, 70), lon=slice(110, 260))  # 110E–100W
    area_sub = areacello.sel(lat=slice(20, 70), lon=slice(110, 260))
    
    # Remove global mean SST anomaly (per member)
    global_mean = SST_ds.weighted(areacello).mean(("lat", "lon"))
    anom = region - global_mean

    # Remove monthly climatology per member
    clim = anom.groupby("time.month").mean("time")
    anom = anom.groupby("time.month") - clim

    # Normalize weights for EOF
    weights = np.sqrt(area_sub / area_sub.max())

    # Loop over members
    PDO_members = []
    for m in anom.member:
        print(m)
        da_m = anom.sel(member=m)
        solver = Eof(da_m, weights=weights)
        eof1 = solver.eofsAsCorrelation(neofs=1)
        eof1.sel(mode=0)
        pc = solver.pcs(npcs=1, pcscaling=1)[:, 0]
        PDO_members.append(
            xr.DataArray(pc, coords={"time": da_m.time, "member": m}, name="PDO_index")
        )

    # Combine all members
    PDO_all = xr.concat(PDO_members, dim="member")
    
    # Standardize (optional, per member)
    PDO_all = (PDO_all - PDO_all.mean("time")) / PDO_all.std("time")

    return PDO_all

#%% Run everything for these models
models = ["EC-Earth3", "MIROC6", "MPI-ESM1-2-LR", "CanESM5", "CESM2", "ACCESS-ESM1-5"]
for model in models:
    print(model)
    #Load in model
    dir = "/archive/depfg/6196306/"
    SST_ds = xr.open_mfdataset(dir+f"CMIP6/{model}/detrend/tos_mon-detrend_{model}_historical_r*i1p1f1_1850-2014_1x1.nc", combine='nested', concat_dim='member').tos
    psl_ds = xr.open_mfdataset(dir+f"CMIP6/{model}/detrend/psl_mon-detrend_{model}_historical_r*i1p1f1_1850-2014_1x1.nc", combine='nested', concat_dim='member').psl
    #z500_ds = xr.open_mfdataset(dir+f"CMIP6/{model}/detrend/z500_mon-detrend_{model}_historical_r*i1p1f1_1850-2014_1x1.nc", combine='nested', concat_dim='member').z500
    #Load in areacello
    areacello = xr.open_dataarray(dir+"CMIP6/areacello.nc")
    #Calculate climate indices
    #El Nino
    nino_12 = nino_index(SST_ds, 12)
    nino_3 = nino_index(SST_ds, 3)
    nino_34 = nino_index(SST_ds, 34)
    nino_4 = nino_index(SST_ds, 4)

    ninos = [nino_12, nino_3, nino_34, nino_4]
    names = ["nino_12", "nino_3", "nino_34", "nino_4"]
    for i in range(len(ninos)):
        save_to_nc(ninos[i], names[i])
        print(f"Saved {names[i]}", flush=True)
    
    #Atlantic nino
    ATL3_box = dict(lat=slice(-3,3), lon=slice(-20,0))
    CABA_box = dict(lat=slice(-20,-10), lon=slice(2,4))
    MEQUE_box = dict(lat=slice(-28,-10), lon=slice(8,15))

    ANi = Atlantic_nino_index(SST_ds, ATL3_box)
    BNi = Atlantic_nino_index(SST_ds, CABA_box)
    Meque_BNi = Atlantic_nino_index(SST_ds, MEQUE_box)

    save_to_nc(ANi, "Atlantic-nino-norm")
    print("Saved ANi", flush=True)
    save_to_nc(BNi, "Benguela-nino-norm")
    print("Saved BNi", flush=True)
    save_to_nc(Meque_BNi, "Benguela-nino-Meque-norm")
    print("Saved Meque_BNi", flush=True)
    
    #Indian Ocean dipole
    IOD = compute_IOD(SST_ds)
    save_to_nc(IOD, "IOD-norm")
    print("Saved IOD", flush=True)
    
    #Subtropical Indian Ocean dipole
    SIOD = compute_SIOD(SST_ds)
    save_to_nc(SIOD, "SIOD-norm")
    print("Saved SIOD", flush=True)
    
    #South Atlantic subtropical dipole
    SASD = compute_SASD(SST_ds)
    save_to_nc(SASD, "SASD-norm")
    print("Saved SASD", flush=True)
    
    #South western indian ocean index
    SWIO = compute_SWIO(SST_ds)
    save_to_nc(SWIO, "SWIO-norm")
    print("Saved SWIO", flush=True)
    
    #Ningaloo nino index 
    NNI = compute_NNI(SST_ds)
    save_to_nc(NNI, "NNI-norm")
    print("Saved NNI", flush=True)
    
    #Tripole pattern index
    TPI = compute_TPI(SST_ds)
    save_to_nc(TPI, "TPI-norm")
    print("Saved TPI", flush=True)
    
    #Pacific Decadal Oscillations
    PDO = compute_PDO(SST_ds, areacello)
    save_to_nc(PDO, "PDO-norm")
    print("Saved PDO", flush=True)
    """
    #Southern annual mode
    SAM = compute_SAM(psl_ds)
    save_to_nc(SAM, "SAM-norm")
    print("Saved SAM", flush=True)
    
    #Botswana high
    BH = compute_BH(psl_ds)
    save_to_nc(BH, "Botswana-high-norm")
    print("Saved BH", flush=True)

    #North Antlantic Oscillation
    NAO = compute_NAO(psl_ds)
    save_to_nc(NAO, "NAO-norm")
    print("Saved NAO", flush=True)
    
    #Southern Oscillation index
    SOI = compute_SOI(psl_ds)
    save_to_nc(SOI, "SOI-norm")
    print("Saved SOI", flush=True)
    
    #Greenland Blocking index
    GBI = compute_GBI(z500_ds)
    save_to_nc(GBI, "GBI-norm")
    print("Saved GBI", flush=True)
    
    #Scandinavian Blocking index
    SBI = compute_SBI(z500_ds)
    save_to_nc(SBI, "SBI-norm")
    print("Saved SBI", flush=True)
    """
    
    #Eofs North Atlantic
    #EOFs = compute_EOFs(psl_ds, areacello)
    #save_to_nc(EOFs[0], "EOF1-norm")
    #save_to_nc(EOFs[1], "EOF2-norm")
    #save_to_nc(EOFs[2], "EOF3-norm")
    #Wave 5
    #wave5 = compute_wave_K(z500_ds, lat_band=50, k=5)
    #save_to_nc(wave5, "wave5-norm")
    #print("Saved wave5")
    
    #Wave 7
    #wave7 = compute_wave_K(z500_ds, lat_band=50, k=7)
    #save_to_nc(wave7, "wave7-norm")
    #print("Saved wave7")
#%%Calculate correlations with SVD patterns
dir = "/your/directory/here/"

models = ["ACCESS-ESM1-5", "CanESM5", " CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]
SVD_tos_midPac_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-midPac-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_tos_midPac_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-midPac-mode0_WEU_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_tos_Npac_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-Npac-mode0_WEU_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_tos_Npac_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-Npac-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_tos_Natl_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-Natl-mode0_WEU_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_tos_ind_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-ind-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_tos_Satl_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-Satl-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_tos_Satl_ATL3_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-ATL3-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_tos_Satl_CABA_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-CABA-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_tos_Satl_MEQUE_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_tos-MEQUE-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_psl_local_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_psl-local-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_psl_global_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_psl-global-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_z500_global_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_z500-global-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_mrso_local_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_mrso-local-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_mrso_global_SA = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_mrso-global-mode0_SA_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_psl_local_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_psl-local-mode0_WEU_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_psl_global_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_psl-global-mode0_WEU_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_z500_NH_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_z500-NH-mode0_WEU_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_z500_global_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_z500-global-mode0_WEU_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_mrso_local_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_mrso-local-mode0_WEU_MMM_historical.nc").assign_coords(model=models).sel(mode=0)
SVD_mrso_global_WEU = xr.open_dataarray(dir+"SVD_output/new_data/detrend/ts0_mrso-global-mode0_WEU_MMM_historical.nc").assign_coords(model=models).sel(mode=0)

SA_SVD = [SVD_psl_local_SA, SVD_psl_global_SA, SVD_z500_global_SA, SVD_tos_Npac_SA, SVD_tos_midPac_SA, SVD_tos_Satl_SA, SVD_tos_Satl_ATL3_SA, SVD_tos_Satl_CABA_SA, SVD_tos_Satl_MEQUE_SA, SVD_tos_ind_SA, SVD_mrso_local_SA, SVD_mrso_global_SA]
SA_SVD_name = ["psl_local_SA", "psl_global_SA", "z500_global_SA", "tos_Npac_SA", "tos_midPac_SA", "tos_Satl_SA", "tos_Satl_ATL3_SA", "tos_Satl_CABA_SA", "tos_Satl_MEQUE_SA", "tos_ind_SA", "mrso_local_SA", "mrso_global_SA"]
WEU_SVD = [SVD_psl_local_WEU, SVD_psl_global_WEU, SVD_z500_NH_WEU, SVD_z500_global_WEU, SVD_tos_Npac_WEU, SVD_tos_midPac_WEU, SVD_tos_Natl_WEU, SVD_mrso_local_WEU, SVD_mrso_global_WEU]
WEU_SVD_name = ["psl_local_WEU", "psl_global_WEU", "z500_NH_WEU", "z500_global_WEU", "tos_Npac_WEU", "tos_midPac_WEU", "tos_Natl_WEU", "mrso_local_WEU", "mrso_global_WEU"]

#Eq. Pacific
nino_12 = xr.open_mfdataset("/Climate_indices/nino_12*.nc", combine='nested', concat_dim='model').tos#..assign_coords(model=models)
nino_3 = xr.open_mfdataset("/Climate_indices/nino_3_*.nc", combine='nested', concat_dim='model').tos#..assign_coords(model=models)
nino_34 = xr.open_mfdataset("/Climate_indices/nino_34*.nc", combine='nested', concat_dim='model').tos#..assign_coords(model=models)
nino_4 = xr.open_mfdataset("/Climate_indices/nino_4*.nc", combine='nested', concat_dim='model').tos#..assign_coords(model=models)

#S. Atlantic
ANi = xr.open_mfdataset("/Climate_indices/Atlantic-nino*.nc", combine='nested', concat_dim='model').__xarray_dataarray_variable__
BNi = xr.open_mfdataset("/Climate_indices/Benguela-nino-norm*.nc", combine='nested', concat_dim='model').__xarray_dataarray_variable__
Meque_BNi = xr.open_mfdataset("/Climate_indices/Benguela-nino-Meque*.nc", combine='nested', concat_dim='model').__xarray_dataarray_variable__
SASD = xr.open_mfdataset("/Climate_indices/SASD-norm*.nc", combine='nested', concat_dim='model').tos#.__xarray_dataarray_variable__

#Indian Ocean
IOD = xr.open_mfdataset("/Climate_indices/IOD-norm*.nc", combine='nested', concat_dim='model').tos
SIOD = xr.open_mfdataset("/Climate_indices/SIOD-norm*.nc", combine='nested', concat_dim='model').tos
SWIO = xr.open_mfdataset("/Climate_indices/SWIO-norm*.nc", combine='nested', concat_dim='model').tos
NNI = xr.open_mfdataset("/Climate_indices/NNI-norm*.nc", combine='nested', concat_dim='model').tos
SWIO2 = xr.open_mfdataset("/Climate_indices/SWIO2-norm*.nc", combine="nested", concat_dim="model").tos
#Others
TPI = xr.open_mfdataset("/Climate_indices/TPI-norm*.nc", combine='nested', concat_dim='model').tos
SAM = xr.open_mfdataset("/Climate_indices/SAM-norm*.nc", combine='nested', concat_dim='model').psl
BH = xr.open_mfdataset("/Climate_indices/BH-norm*.nc", combine='nested', concat_dim='model').psl
NAO = xr.open_mfdataset("/Climate_indices/NAO-norm*.nc", combine='nested', concat_dim='model').psl
SOI = xr.open_mfdataset("/Climate_indices/SOI-norm*.nc", combine='nested', concat_dim='model').psl
PDO = xr.open_mfdataset("/Climate_indices/PDO-norm*.nc", combine='nested', concat_dim='model').PDO_index
GBI = xr.open_mfdataset("/Climate_indices/GBI-norm*.nc", combine='nested', concat_dim='model').GBI
SBI = xr.open_mfdataset("/Climate_indices/SBI-norm*.nc", combine='nested', concat_dim='model').SCA_index

EOF1 = xr.open_mfdataset("/Climate_indices/EOF1-norm*.nc", combine='nested', concat_dim='model').EOF1
EOF2 = xr.open_mfdataset("/Climate_indices/EOF2-norm*.nc", combine='nested', concat_dim='model').EOF2
EOF3 = xr.open_mfdataset("/Climate_indices/EOF3-norm*.nc", combine='nested', concat_dim='model').EOF3

#Calculate correlations
for i in range(len(WEU_SVD)):
    print(WEU_SVD_name[i])
    print("Eq. Pacific correlations")
    print(f"Nino12    {xr.corr(WEU_SVD[i], nino_12).values:.2f}")
    print(f"Nino3     {xr.corr(WEU_SVD[i], nino_3).values:.2f}")
    print(f"Nino34    {xr.corr(WEU_SVD[i], nino_34).values:.2f}")
    print(f"Nino4     {xr.corr(WEU_SVD[i], nino_4).values:.2f}")

for i in range(len(WEU_SVD)):
    print(WEU_SVD_name[i])
    print("S. Atlantic correlations")
    print(f"ANi       {xr.corr(WEU_SVD[i], ANi).values:.2f}")
    print(f"BNi       {xr.corr(WEU_SVD[i], BNi).values:.2f}")
    print(f"BNi-Meque {xr.corr(WEU_SVD[i], Meque_BNi).values:.2f}")
    print(f"SASD      {xr.corr(WEU_SVD[i], SASD).values:.2f}")

    print("Indian Ocean correlations")
    print(f"IOD       {xr.corr(WEU_SVD[i], IOD).values:.2f}")
    print(f"SIOD      {xr.corr(WEU_SVD[i], SIOD).values:.2f}")
    print(f"NNi       {xr.corr(WEU_SVD[i], NNI).values:.2f}")
    print(f"SWIO      {xr.corr(WEU_SVD[i], SWIO).values:.2f}")

for i in range(len(WEU_SVD)):
    print(WEU_SVD_name[i])
    print(f"TPi       {xr.corr(WEU_SVD[i], TPI).values:.2f}")
    print(f"SAM       {xr.corr(WEU_SVD[i], SAM).values:.2f}")
    print(f"BH        {xr.corr(WEU_SVD[i], BH).values:.2f}")
    #print(f"NAO       {xr.corr(SA_SVD[i], NAO).values:.2f}")
    print(f"SOI       {xr.corr(WEU_SVD[i], SOI).values:.2f}")
    print(f"PDO       {xr.corr(WEU_SVD[i], PDO).values:.2f}")
    print(f"GBI       {xr.corr(WEU_SVD[i], GBI).values:.2f}")
    print(f"SBI       {xr.corr(WEU_SVD[i], SBI).values:.2f}")

for i in range(len(WEU_SVD)):
    print(WEU_SVD_name[i])
    print(f"EOF1       {xr.corr(WEU_SVD[i], EOF1).values:.2f}")
    print(f"EOF2       {xr.corr(WEU_SVD[i], EOF2).values:.2f}")
    #print(f"EOF3       {xr.corr(SA_SVD[i], EOF3).values:.2f}")

#%% Try again but with optimal lag
lags = np.arange(-12, 13)

def lag_corr(SVD, var, lags):
    # stack model and member into one dimension
    SVDs = SVD.stack(sample=("model", "member"))
    vars = var.stack(sample=("model", "member"))

    corrs = []

    for lag in lags:
        shifted = vars.shift(time=lag)
        r = xr.corr(SVDs, shifted, dim="time")
        corrs.append(r)

    corrs = xr.concat(corrs, dim="lag")
    corrs["lag"] = lags

    # single number per lag (no remaining dims)
    corrs_mean = corrs.mean("sample")

    best_lag = corrs_mean.lag[np.nanargmax(np.abs(corrs_mean))]

    print(
        f"Highest |r| = {float(np.abs(corrs_mean).max()):.2f} "
        f"at lag {int(best_lag.values)}"
    )

    return corrs_mean, best_lag


for i in range(len(WEU_SVD)):
    print(WEU_SVD_name[i])
    print("Eq. Pacific correlations")
    lag_corr(WEU_SVD[i], nino_12, lags)
    lag_corr(WEU_SVD[i], nino_3, lags)
    lag_corr(WEU_SVD[i], nino_34, lags)
    lag_corr(WEU_SVD[i], nino_4, lags)
    
for i in range(len(WEU_SVD)):
    print(WEU_SVD_name[i])
    print("S. Atlantic correlations")
    print("ANi")
    lag_corr(WEU_SVD[i], ANi, lags)
    print("BNi")
    lag_corr(WEU_SVD[i], BNi, lags)
    print("BNi-Meque")
    lag_corr(WEU_SVD[i], Meque_BNi, lags)
    print("SASD")
    lag_corr(WEU_SVD[i], SASD, lags)

    print("Indian Ocean correlations")
    print("IOD")
    lag_corr(WEU_SVD[i], IOD, lags)
    print("SIOD")
    lag_corr(WEU_SVD[i], SIOD, lags)
    print("NNi")
    lag_corr(WEU_SVD[i], NNI, lags)
    print("SWIO")
    lag_corr(SA_SVD[i], SWIO, lags)
    
for i in range(len(SA_SVD)):
    print(SA_SVD_name[i])
    print("BH")
    lag_corr(SA_SVD[i], BH, lags)
    print("TPi")
    lag_corr(WEU_SVD[i], TPI, lags)
    print("SAM")
    lag_corr(WEU_SVD[i], SAM, lags)
    print("SOI")
    lag_corr(WEU_SVD[i], SOI, lags)
    print("PDO")
    lag_corr(WEU_SVD[i], PDO, lags)
    print("GBI")
    lag_corr(WEU_SVD[i], GBI, lags)
    print("SBI")
    lag_corr(WEU_SVD[i], SBI, lags)
    print("EOF1")
    lag_corr(WEU_SVD[i], EOF1, lags)
    print("EOF2")
    lag_corr(WEU_SVD[i], EOF2, lags)

#%% Also calculate the correlations of al of them with local SPEI-12
models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]

SPEI_WEU_list = []
SPEI_SA_list = []
for model in models:
    spei_WEU_model = xr.open_mfdataset(dir+f"SPEI/{model}/new_data/1950-2014/SPEI12_monthly_1950_2014_r*_{model}_WEU.nc", combine="nested", concat_dim="member").__xarray_dataarray_variable__.expand_dims(model=[model])
    SPEI_WEU_list.append(spei_WEU_model)
    spei_SA_model = xr.open_mfdataset(dir+f"SPEI/{model}/new_data/1950-2014/SPEI12_monthly_1950_2014_r*_{model}_SA.nc", combine="nested", concat_dim="member").__xarray_dataarray_variable__.expand_dims(model=[model])
    SPEI_SA_list.append(spei_SA_model)
SPEI_WEU = xr.concat(SPEI_WEU_list, dim="model", coords="minimal")
SPEI_SA = xr.concat(SPEI_SA_list, dim="model", coords="minimal")

indices = [nino_12, nino_3, nino_34, nino_4, ANi, BNi, Meque_BNi, SASD, IOD, SIOD, NNI, SWIO, PDO, 
           TPI, SAM, BH, SOI, GBI, SBI, EOF1, EOF2, EOF3]
names = ["nino_12", "nino_3", "nino_34", "nino_4", "ANi", "BNi", "Meque_BNi", "SASD", "IOD", "SIOD", "NNI", "SWIO", "PDO", 
           "TPI", "SAM", "BH", "SOI", "GBI", "SBI", "EOF1", "EOF2", "EOF3"]

for i in range(len(indices)):
    print(names[i], "WEU")
    lag_corr(SPEI_WEU, indices[i], lags)
    print(names[i], "SA")
    lag_corr(SPEI_SA, indices[i], lags)
