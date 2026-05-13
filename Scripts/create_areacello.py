#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:32:55 2025

@author: Jonna van Mourik
"""

import numpy as np
import xarray as xr

# Earth's radius in meters
R = 6_371_000  

# Define grid parameters from gridfile.txt
xsize, ysize = 360, 180
xfirst, xinc = -179.5, 1.0
yfirst, yinc = -89.5, 1.0

# Generate latitude and longitude coordinates
lons = np.linspace(xfirst, xfirst + (xsize - 1) * xinc, xsize)
lats = np.linspace(yfirst, yfirst + (ysize - 1) * yinc, ysize)

# Convert to radians
dlat = np.radians(yinc)
dlon = np.radians(xinc)

# Compute grid cell areas (latitude-dependent)
areas = np.zeros((ysize, xsize))

for i, lat in enumerate(lats):
    dy = R * dlat  # North-south distance
    dx = R * dlon * np.cos(np.radians(lat))  # East-west distance at latitude
    areas[i, :] = dx * dy  # Assign area to all longitudes

# Convert to xarray DataArray
areacello = xr.DataArray(
    areas, dims=["lat", "lon"], coords={"lat": lats, "lon": lons}
)
areacello.attrs["units"] = "m2"
areacello.attrs["long_name"] = "Grid cell area"

# Save as NetCDF
areacello.to_netcdf("areacello.nc")

print("areacello.nc file created successfully!")
