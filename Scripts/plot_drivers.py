#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 10:05:13 2025

@author: Jonna van Mourik
Make figures of tos, psl, z500 and mrso together for WEU and SA
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
#import data
dir = "/your/directory/here/"
mrso_WEU = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_mrso_global_SPEI_WEU.nc").U_mode0_mrso_global
mrso_SA = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_mrso_global_SPEI_SA.nc").U_mode0_mrso_global
psl_WEU = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_psl_global_SPEI_WEU.nc").U_mode0_psl_global
psl_SA = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_psl_global_SPEI_SA.nc").U_mode0_psl_global
z500_WEU = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_z500_global_SPEI_WEU.nc").U_mode0_z500_global
z500_SA = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_z500_global_SPEI_SA.nc").U_mode0_z500_global
tos_WEU = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_tos_global_SPEI_WEU.nc").U_mode0_tos_global
tos_SA = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_tos_global_SPEI_SA.nc").U_mode0_tos_global

#Select regions of interest
ts_name_WEU = ["tos, global", "tos, Natl", "tos, midPac", "tos, Npac", "mrso", "psl, global", "psl, local", "z500, local", "z500, global"]
colours_WEU = ["blue", "#084594","dodgerblue", "#41b6c4", "orchid", "#fe9929", "#F0B400",  "indianred", "firebrick"]
ts_name_SA = ["tos, global", "tos, Satl", "tos, ind", "tos, midPac", "tos, Npac", "mrso", "psl, global", "psl, local", "z500, global"]
colours_SA = ["blue", "#084594", "cyan","dodgerblue", "#41b6c4", "orchid", "#fe9929", "#F0B400", "firebrick"]

tos_midPac_patch = mpatches.Rectangle(xy=[-180, -20], width=100, height=40, transform=ccrs.PlateCarree(), color=colours_SA[3], fill=False, linewidth=2, zorder=10)
tos_Npac1_patch = mpatches.Rectangle(xy=[-180, 20], width=60, height=40, transform=ccrs.PlateCarree(), color=colours_SA[4], fill=False, linewidth=2, zorder=10)
tos_Npac2_patch = mpatches.Rectangle(xy=[130, 20], width=50, height=40, transform=ccrs.PlateCarree(), color=colours_SA[4], fill=False, linewidth=2, zorder=10)
tos_Natl_patch = mpatches.Rectangle(xy=[-80, 20], width=120, height=60, transform=ccrs.PlateCarree(), color=colours_WEU[1], fill=False, linewidth=2, zorder=10)
tos_Satl_patch = mpatches.Rectangle(xy=[-40, -60], width=60, height=80, transform=ccrs.PlateCarree(), color=colours_SA[1], fill=False, linewidth=2, zorder=10)
tos_ind_patch = mpatches.Rectangle(xy=[40, -40], width=100, height=60, transform=ccrs.PlateCarree(), color=colours_SA[2], fill=False, linewidth=2, zorder=10)
mrso_local_SA_patch = mpatches.Rectangle(xy=[12, -35], width=25, height=25, transform=ccrs.PlateCarree(), color=colours_SA[5], fill=False, linewidth=2, zorder=10)
mrso_local_WEU_patch = mpatches.Rectangle(xy=[-5, 42], width=22, height=16, transform=ccrs.PlateCarree(), color=colours_WEU[5], fill=False, linewidth=2, zorder=10)
psl_local_SA_patch = mpatches.Rectangle(xy=[12, -35], width=25, height=25, transform=ccrs.PlateCarree(), color=colours_SA[7], fill=False, linewidth=2, zorder=10)
psl_local_WEU_patch = mpatches.Rectangle(xy=[-80, 20], width=120, height=60, transform=ccrs.PlateCarree(), color=colours_WEU[7], fill=False, linewidth=2, zorder=10)
z500_NH_patch = mpatches.Rectangle(xy=[30, -180], width=360, height=35, transform=ccrs.PlateCarree(), color=colours_WEU[8], fill=False, linewidth=2, zorder=10)
    
#masks
mask_WEU = xr.open_dataset(dir+"masks/1x1/mask_WEU_1x1.nc").Band1
mask_SA = xr.open_dataset(dir+"masks/1x1/mask_SA_1x1.nc").Band1

cmap_mrso = "BrBG"
cmap_rest = "RdBu_r"#"coolwarm"
proj = ccrs.EqualEarth() #or ccrs.PlateCarree()
transf = ccrs.PlateCarree()

#%% Plot figure for WEU

fig, ax = plt.subplots(2,2, figsize=(12,6), subplot_kw={"projection": proj}, sharey=True)
psl_WEU.plot(ax=ax[0,0], cmap=cmap_rest, transform=transf, vmin=-0.03, vmax=0.03, add_colorbar=False)
#add local psl
ax[0,0].add_patch(mpatches.Rectangle(xy=[-80, 20], width=120, height=60, transform=transf, color=colours_WEU[6], fill=False, linewidth=2, zorder=2))
ax[0,0].set_title("a) Sea level pressure")
#ax[0,0].set_title("a)", loc="left")
ax[0,0].text(-8, 24, "psl-local", color="black", transform=transf)
im_z500_WEU = z500_WEU.plot(ax=ax[0,1], cmap=cmap_rest, transform=transf, vmin=-0.03, vmax=0.03, add_colorbar=False)
#add NH z500
ax[0,1].add_patch(mpatches.Rectangle(xy=[-180, 30], width=360, height=35, transform=transf, color=colours_WEU[7], fill=False, linewidth=2, zorder=2))
ax[0,1].set_title("b) Geopotential height 500 hPa")
#ax[0,1].set_title("b)", loc="left")
ax[0,1].text(59, 34, "z500-NH", color="black", transform=transf)
tos_WEU.plot(ax=ax[1,0], cmap=cmap_rest, transform=transf, vmin=-0.03, vmax=0.03, add_colorbar=False)
#add n-atl tos
ax[1,0].add_patch(mpatches.Rectangle(xy=[-80, 20], width=120, height=60, transform=transf, color=colours_WEU[1], fill=False, linewidth=2, zorder=2))
#add N-pac tos
ax[1,0].add_patch(mpatches.Rectangle(xy=[-180, 20], width=60, height=40, transform=transf, color=colours_SA[4], fill=False, linewidth=2, zorder=2))
ax[1,0].add_patch(mpatches.Rectangle(xy=[130, 20], width=50, height=40, transform=transf, color=colours_SA[4], fill=False, linewidth=2, zorder=2))
#add mid-pac tos
ax[1,0].add_patch(mpatches.Rectangle(xy=[-180, -20], width=100, height=40, transform=transf, color=colours_SA[3], fill=False, linewidth=2, zorder=2))
ax[1,0].set_title("c) Sea surface temperature")
#ax[1,0].set_title("c)", loc="left")
ax[1,0].text(3, 24, "N-Atl", color="black", transform=transf)
ax[1,0].text(-175, 24, "N-Pac", color="black", transform=transf)
ax[1,0].text(-118, -17, "Eq-Pac", color="black", transform=transf)
im_mrso_WEU = (mrso_WEU*-1).plot(ax=ax[1,1], cmap=cmap_mrso, transform=transf, vmin=-0.03, vmax=0.03, add_colorbar=False)
#add local mrso
ax[1,1].add_patch(mpatches.Rectangle(xy=[-5, 42], width=22, height=16, transform=transf, color=colours_WEU[4], fill=False, linewidth=2, zorder=2))
ax[1,1].set_title("d) Total soil moisture content")
#ax[1,1].set_title("d)", loc="left")
#Add colourbar
fig.subplots_adjust(right=0.8)
pos_left = ax[1,0].get_position()   # Bottom-left subplot
pos_right = ax[1,1].get_position()  # Bottom-right subplot
#cbar_ax_z500 = fig.add_axes([0.9, 0.53, 0.02, 0.44])
cbar_ax_z500 = fig.add_axes([pos_left.x0-0.05, pos_left.y0-0.16, pos_left.width, 0.02])
#cbar_ax_mrso = fig.add_axes([0.9, 0.04, 0.02, 0.44])
cbar_ax_mrso = fig.add_axes([pos_right.x0+0.03, pos_right.y0-0.16, pos_right.width, 0.02])
fig.colorbar(im_z500_WEU, cax=cbar_ax_z500, orientation="horizontal")
fig.colorbar(im_mrso_WEU, cax=cbar_ax_mrso, orientation="horizontal")

for axes in ax.flatten():
    axes.coastlines(linewidth=0.5, zorder=1)
    #axes.axis('off')
    #axes.set_extent([-170, 170, -63, 90], crs=ccrs.PlateCarree())
    if axes==ax[1,1]:
        axes.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='white', linewidths=1, transform=transf)
    else:
        axes.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='black', linewidths=1, transform=transf)
    
#plt.tight_layout(rect=[0, 0, 0.9, 1.04]) 
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()
filename = "Figure_4"
fig.savefig(f"{filename}.jpg", dpi=1200, bbox_inches="tight")
fig.savefig(f"{filename}.pdf", bbox_inches="tight")

#%% Same but for SA

fig, ax = plt.subplots(2,2, figsize=(12,6), subplot_kw={"projection": proj}, sharey=True)
(psl_SA*-1).plot(ax=ax[0,0], cmap=cmap_rest, transform=ccrs.PlateCarree(), vmin=-0.03, vmax=0.03, add_colorbar=False)
#add local psl
ax[0,0].add_patch(mpatches.Rectangle(xy=[12, -35], width=25, height=25, transform=ccrs.PlateCarree(), color=colours_SA[6], fill=False, linewidth=2, zorder=2))
ax[0,0].set_title("a) Sea level pressure")
#ax[0,0].set_title("a)", loc="left")
ax[0,0].text(2, -43, "psl-local", color="black", transform=transf)
im_z500_SA = z500_SA.plot(ax=ax[0,1], cmap=cmap_rest, transform=ccrs.PlateCarree(), vmin=-0.03, vmax=0.03, add_colorbar=False)
ax[0,1].set_title("b) Geopotential height 500 hPa")
#ax[0,1].set_title("b)", loc="left")
tos_SA.plot(ax=ax[1,0], cmap=cmap_rest, transform=ccrs.PlateCarree(), vmin=-0.03, vmax=0.03, add_colorbar=False)
#add s-atl tos
ax[1,0].add_patch(mpatches.Rectangle(xy=[-40, -60], width=60, height=80, transform=ccrs.PlateCarree(), color=colours_SA[1], fill=False, linewidth=2, zorder=2))
#add N-pac tos
ax[1,0].add_patch(mpatches.Rectangle(xy=[-180, 20], width=60, height=40, transform=ccrs.PlateCarree(), color=colours_SA[4], fill=False, linewidth=2, zorder=2))
ax[1,0].add_patch(mpatches.Rectangle(xy=[130, 20], width=50, height=40, transform=ccrs.PlateCarree(), color=colours_SA[4], fill=False, linewidth=2, zorder=2))
#add mid-pac tos
ax[1,0].add_patch(mpatches.Rectangle(xy=[-180, -20], width=100, height=40, transform=ccrs.PlateCarree(), color=colours_SA[3], fill=False, linewidth=2, zorder=2))
#add ind tos
ax[1,0].add_patch(mpatches.Rectangle(xy=[40, -40], width=100, height=60, transform=ccrs.PlateCarree(), color=colours_SA[2], fill=False, linewidth=2, zorder=2))
ax[1,0].set_title("c) Sea surface temperature")
#ax[1,0].set_title("c)", loc="left")
ax[1,0].text(-175, 24, "N-Pac", color="black", transform=transf)
ax[1,0].text(-118, -17, "Eq-Pac", color="black", transform=transf)
ax[1,0].text(-12, -55, "S-Atl", color="black", transform=transf)
ax[1,0].text(45, -35, "Ind", color="black", transform=transf)


im_mrso_SA = (mrso_SA*-1).plot(ax=ax[1,1], cmap=cmap_mrso, transform=ccrs.PlateCarree(), vmin=-0.03, vmax=0.03, add_colorbar=False)
#add local mrso
ax[1,1].add_patch(mpatches.Rectangle(xy=[12, -35], width=25, height=25, transform=ccrs.PlateCarree(), color="orchid", fill=False, linewidth=2, zorder=2))
ax[1,1].set_title("d) Total soil moisture content")
#ax[1,1].set_title("d)", loc="left")

#Add colourbar
fig.subplots_adjust(right=0.8)
pos_left = ax[1,0].get_position()   # Bottom-left subplot
pos_right = ax[1,1].get_position()  # Bottom-right subplot
cbar_ax_z500 = fig.add_axes([pos_left.x0-0.05, pos_left.y0-0.16, pos_left.width, 0.02])
#cbar_ax_z500 = fig.add_axes([0.9, 0.53, 0.02, 0.44])
cbar_ax_mrso = fig.add_axes([pos_right.x0+0.03, pos_right.y0-0.16, pos_right.width, 0.02])
#cbar_ax_mrso = fig.add_axes([0.9, 0.04, 0.02, 0.44])
fig.colorbar(im_z500_SA, cax=cbar_ax_z500, orientation="horizontal")
fig.colorbar(im_mrso_SA, cax=cbar_ax_mrso, orientation="horizontal")

for axes in ax.flatten():
    axes.coastlines(linewidth=0.5, zorder=1)
    #axes.set_extent([-180, 180, -63, 90], crs=ccrs.PlateCarree())
    if axes==ax[1,1]:
        axes.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='white', linewidths=1, transform=ccrs.PlateCarree())
    else:
        axes.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='black', linewidths=1, transform=ccrs.PlateCarree())
    
plt.tight_layout(rect=[0, 0, 0.9, 1]) 
plt.show()
filename = "Figure_3"
fig.savefig(f"{filename}.jpg", dpi=1200, bbox_inches="tight")
fig.savefig(f"{filename}.pdf", bbox_inches="tight")

#%% Make local figures for the drivers
psl_local_WEU = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_psl_local_SPEI_WEU.nc").U_mode0_psl_local
Npac_WEU = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_tos_Npac_SPEI_WEU.nc").U_mode0_tos_Npac
EqPac_WEU = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_tos_midPac_SPEI_WEU.nc").U_mode0_tos_midPac
Natl_WEU = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_tos_Natl_SPEI_WEU.nc").U_mode0_tos_Natl
z500_NH_WEU = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_z500_NH_SPEI_WEU.nc").U_mode0_z500_NH
mrso_local_WEU = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_mrso_local_SPEI_WEU.nc").U_mode0_mrso_local

psl_local_SA = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_psl_local_SPEI_SA.nc").U_mode0_psl_local
Npac_SA = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_tos_Npac_SPEI_SA.nc").U_mode0_tos_Npac
EqPac_SA = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_tos_midPac_SPEI_SA.nc").U_mode0_tos_midPac
Satl_SA = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_tos_Satl_SPEI_SA.nc").U_mode0_tos_Satl
Ind_SA = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_tos_ind_SPEI_SA.nc").U_mode0_tos_ind
mrso_local_SA = xr.open_dataset(dir+"SVD_output/new_data/detrend/SVD_mode0_lag+0_mrso_local_SPEI_SA.nc").U_mode0_mrso_local

#Make plots
#%% For Southern Africa
fig = plt.figure(figsize=(12,6))
fig.suptitle("Southern Africa", fontsize=18)

# First subplot
ax0 = fig.add_subplot(2,3,1, projection=proj)
(-1*psl_local_SA).plot(ax=ax0, transform=transf, cmap=cmap_rest, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax0.add_patch(mpatches.Rectangle(
    xy=[12, -35], width=25, height=25, 
    transform=ccrs.PlateCarree(), color=colours_SA[6], fill=False, linewidth=2, zorder=2))
ax0.set_extent([11, 38, -36, -9])
ax0.set_title("a) Sea level pressure, local")

# Second subplot: centered on 180° longitude
proj_180 = ccrs.EqualEarth(central_longitude=-165)
ax1 = fig.add_subplot(2,3,2, projection=proj_180)
Npac_SA.plot(ax=ax1, transform=transf, cmap=cmap_rest, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax1.add_patch(mpatches.Rectangle(xy=[130, 20], width=110, height=40, transform=ccrs.PlateCarree(), color=colours_SA[4], fill=False, linewidth=2, zorder=2))
ax1.add_feature(cfeature.LAND, color="white", zorder=1)
ax1.set_extent([-115, 126, 18, 45])
ax1.set_title("b) Sea surface temperature, Npac")

#EqPac
ax2 = fig.add_subplot(2,3,3, projection=proj_180)
tos = EqPac_SA.plot(ax=ax2, transform=transf, cmap=cmap_rest, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax2.add_patch(mpatches.Rectangle(xy=[-180, -20], width=100, height=40, transform=ccrs.PlateCarree(), color=colours_SA[3], fill=False, linewidth=2, zorder=2))
ax2.add_feature(cfeature.LAND, color="white", zorder=1)
ax2.set_extent([-182, -78, -16, 16])
ax2.set_title("c) Sea surface temperature, EqPac")

#Satl
ax3 = fig.add_subplot(2,3,4, projection=proj)
Satl_SA.plot(ax=ax3, transform=transf, cmap=cmap_rest, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax3.add_patch(mpatches.Rectangle(xy=[-40, -60], width=60, height=80, transform=ccrs.PlateCarree(), color=colours_SA[1], fill=False, linewidth=2, zorder=2))
ax3.add_feature(cfeature.LAND, color="white", zorder=1)
ax3.set_extent([-42, 20, -60, 20])
ax3.set_title("d) Sea surface temperature, Satl")

#Ind
ax4 = fig.add_subplot(2,3,5, projection=proj)
Ind_SA.plot(ax=ax4, transform=transf, cmap=cmap_rest, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax4.add_patch(mpatches.Rectangle(xy=[40, -40], width=100, height=60, transform=ccrs.PlateCarree(), color=colours_SA[2], fill=False, linewidth=2, zorder=2))
ax4.add_feature(cfeature.LAND, color="white", zorder=1)
ax4.set_extent([37, 142, -30, 15])
ax4.set_title("e) Sea surface temperature, Ind")

#mrso
ax5 = fig.add_subplot(2,3,6, projection=proj)
mrso = (mrso_local_SA*-1).plot(ax=ax5, transform=transf, cmap=cmap_mrso, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax5.add_patch(mpatches.Rectangle(xy=[12, -35], width=25, height=25, transform=ccrs.PlateCarree(), color="orchid", fill=False, linewidth=2, zorder=2))
ax5.set_extent([11, 38, -36, -9])
ax5.set_title("f) Total soil moisture content, local")

for axes in [ax0, ax1, ax2, ax3, ax4, ax5]:
    axes.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors="white", linewidths=1, transform=transf)
    axes.coastlines()
    
fig.subplots_adjust(right=0.8)
pos_left = ax2.get_position()   # Bottom-left subplot
pos_right = ax5.get_position()  # Bottom-right subplot
cbar_ax_z500 = fig.add_axes([pos_left.x0+0.3, pos_left.y0-0.08, 0.01, 0.36])
#cbar_ax_z500 = fig.add_axes([0.9, 0.53, 0.02, 0.44])
cbar_ax_mrso = fig.add_axes([pos_right.x0+0.27, pos_right.y0-0.08, 0.01, 0.36])
#cbar_ax_mrso = fig.add_axes([0.9, 0.04, 0.02, 0.44])
fig.colorbar(tos, cax=cbar_ax_z500, orientation="vertical")
fig.colorbar(mrso, cax=cbar_ax_mrso, orientation="vertical")

#Set x and y lims

plt.tight_layout(rect=[0, 0, 0.9, 1]) 
filename = "Figure_S1"
fig.savefig(f"{filename}.jpg", dpi=1200, bbox_inches="tight")
fig.savefig(f"{filename}.pdf", bbox_inches="tight")

#%% and WEU
fig = plt.figure(figsize=(12,6))
fig.suptitle("Western Europe", fontsize=18)

#psl
ax0 = fig.add_subplot(2,3,1, projection=proj)
(psl_local_WEU).plot(ax=ax0, transform=transf, cmap=cmap_rest, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax0.add_patch(mpatches.Rectangle(xy=[-80, 20], width=120, height=60, transform=transf, color=colours_WEU[6], fill=False, linewidth=2, zorder=2))
ax0.set_extent([-80, 42, 18, 85])
ax0.set_title("a) Sea level pressure, local")

# NH
proj_180 = ccrs.EqualEarth(central_longitude=-165)
ax1 = fig.add_subplot(2,3,2, projection=proj)
z500_NH_WEU.plot(ax=ax1, transform=transf, cmap=cmap_rest, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax1.add_patch(mpatches.Rectangle(xy=[-180, 30], width=360, height=35, transform=transf, color=colours_WEU[7], fill=False, linewidth=2, zorder=2))
ax1.set_extent([-179, 179, 25, 80])
ax1.set_title("b) Geopotential height 500 hPa, NH")

#Npac
ax2 = fig.add_subplot(2,3,3, projection=proj_180)
Npac_WEU.plot(ax=ax2, transform=transf, cmap=cmap_rest, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax2.add_patch(mpatches.Rectangle(xy=[130, 20], width=110, height=40, transform=ccrs.PlateCarree(), color=colours_WEU[4], fill=False, linewidth=2, zorder=2))
ax2.add_feature(cfeature.LAND, color="white", zorder=1)
ax2.set_extent([-115, 126, 18, 45])
ax2.set_title("c) Sea surface temperature, Npac")


#EqPac
ax3 = fig.add_subplot(2,3,4, projection=proj_180)
tos = (-1*EqPac_WEU).plot(ax=ax3, transform=transf, cmap=cmap_rest, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax3.add_patch(mpatches.Rectangle(xy=[-180, -20], width=100, height=40, transform=ccrs.PlateCarree(), color=colours_WEU[3], fill=False, linewidth=2, zorder=2))
ax3.add_feature(cfeature.LAND, color="white", zorder=1)
ax3.set_extent([-182, -78, -16, 16])
ax3.set_title("d) Sea surface temperature, EqPac")

#Natl
ax4 = fig.add_subplot(2,3,5, projection=proj)
Natl_WEU.plot(ax=ax4, transform=transf, cmap=cmap_rest, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax4.add_patch(mpatches.Rectangle(xy=[-80, 20], width=120, height=60, transform=transf, color=colours_WEU[1], fill=False, linewidth=2, zorder=2))
ax4.add_feature(cfeature.LAND, color="white", zorder=1)
ax4.set_extent([-80, 42, 18, 85])
ax4.set_title("e) Sea surface temperature, Natl")

#mrso
ax5 = fig.add_subplot(2,3,6, projection=proj)
mrso = (mrso_local_WEU*-1).plot(ax=ax5, transform=transf, cmap=cmap_mrso, vmin=-0.03, vmax=0.03, add_colorbar=False)
ax5.add_patch(mpatches.Rectangle(xy=[-5, 42], width=22, height=16, transform=transf, color=colours_WEU[4], fill=False, linewidth=2, zorder=2))
ax5.set_extent([-6, 18, 41, 59])
ax5.set_title("f) Total soil moisture content, local")

for axes in [ax0, ax1, ax2, ax3, ax4, ax5]:
    axes.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors="white", linewidths=1, transform=transf)
    axes.coastlines()
    
fig.subplots_adjust(right=0.8)
pos_left = ax2.get_position()   # Bottom-left subplot
pos_right = ax5.get_position()  # Bottom-right subplot
cbar_ax_z500 = fig.add_axes([pos_left.x0+0.3, pos_left.y0-0.11, 0.01, 0.36])
#cbar_ax_z500 = fig.add_axes([0.9, 0.53, 0.02, 0.44])
cbar_ax_mrso = fig.add_axes([pos_right.x0+0.3, pos_right.y0-0.06, 0.01, 0.36])
#cbar_ax_mrso = fig.add_axes([0.9, 0.04, 0.02, 0.44])
fig.colorbar(tos, cax=cbar_ax_z500, orientation="vertical")
fig.colorbar(mrso, cax=cbar_ax_mrso, orientation="vertical")

#Set x and y lims

plt.tight_layout(rect=[0, 0, 0.9, 1]) 
filename = "Figure_S2"
fig.savefig(f"{filename}.jpg", dpi=1200, bbox_inches="tight")
fig.savefig(f"{filename}.pdf", bbox_inches="tight")