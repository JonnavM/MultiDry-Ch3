#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 10:30:32 2025

@author: 6196306
Simple figure with SA and WEU, including climatology
"""
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from cartopy.feature import NaturalEarthFeature
import matplotlib.patches as mpatches


#Import data
dir = "/your/directory/here/"
models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]

#masks for models
mask_WEU = xr.open_dataarray(dir+"masks/1x1/mask_WEU_1x1.nc")
mask_SA = xr.open_dataarray(dir+"masks/1x1/mask_SA_1x1.nc")

#PR and PET for SA and WEU
PET_WEU = xr.open_dataarray(dir+"PET-WEU_MM.nc")
PR_WEU = xr.open_dataarray(dir+"PR-WEU_MM.nc")*3600*24
PET_SA = xr.open_dataarray(dir+"PET-SA_MM.nc")
PR_SA = xr.open_dataarray(dir+"PR-SA_MM.nc")*3600*24

#Change units to mm/month
def mmpermonth(da):
    days_in_month = da.time.dt.days_in_month
    da_month = da*days_in_month
    da_month.attrs['units'] = 'mm/month'
    return da_month

pet_WEU = mmpermonth(PET_WEU)
pr_WEU = mmpermonth(PR_WEU)
pet_SA = mmpermonth(PET_SA)
pr_SA = mmpermonth(PR_SA)

pr_list_reg = [pr_WEU, pr_SA]
pet_list_reg = [pet_WEU, pet_SA]

# Use Natural Earth "rivers_lake_centerlines" at high resolution (10m)
rivers_10m = NaturalEarthFeature(
    category="physical",
    name="rivers_lake_centerlines",
    scale="10m",
    facecolor="none"
)


#%% Make plot
fig, ax = plt.subplots(1, figsize=(12,6), subplot_kw={"projection": ccrs.EqualEarth()})
ax.stock_img()
#ax.add_feature(cfeature.RIVERS)
#ax.add_feature(cfeature.LAKES)
ax.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='black', linewidths=1, transform=ccrs.PlateCarree())
ax.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='black', linewidths=1, transform=ccrs.PlateCarree())
ax.set_extent([-20, 55, -45, 75], crs=ccrs.PlateCarree())
#add inlets
reg = ["WEU","SA"]
month_letter = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

for i, region_name in enumerate(reg):
    if region_name == "WEU":
        reg_lat = slice(45, 55)
        reg_lon = slice(0, 13)
        left = 0.25
        bottom = 0.55
        inset_center_x = 1
        inset_center_y = 50
        box_center_lon = reg_lon.start - 30
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
        title = "Western Europe"
    elif region_name == "SA":
        reg_lat = slice(-33, -21)
        reg_lon = slice(15, 31)
        left = 0.25
        bottom = 0.15
        inset_center_x = 17
        inset_center_y = -27
        box_center_lon = reg_lon.start - 42
        box_center_lat = (reg_lat.start + reg_lat.stop) / 2
        title = "Southern Africa"

    fig.patch.set_facecolor('white')
    left, bottom, width, height = [left+0.375*2, bottom, 0.15, 0.3]
    ax_inset = fig.add_axes([left-0.16, bottom+0.01, width, height])
    ax_inset.set_facecolor('white')
    
    pr_reg = pr_list_reg[i].mean(dim=("model", "member")).groupby("time.month").mean("time")
    pr_reg_std = pr_list_reg[i].mean(dim=("model", "member")).groupby("time.month").std("time")
    pet_reg = pet_list_reg[i].mean(dim=("model", "member")).groupby("time.month").mean("time")
    pet_reg_std = pet_list_reg[i].mean(dim=("model", "member")).groupby("time.month").std("time")

    ax_inset.bar(pr_reg.month, pr_reg, color="tab:blue", yerr=pr_reg_std, label="Pr")
    ax_inset.set_xticks(pr_reg.month, month_letter, fontsize=8)

    ax_inset.set_ylim(0,200)
    ax_inset.tick_params(axis='y', labelsize=8)
    ax_inset.yaxis.tick_right()
    ax_inset.yaxis.set_label_position("right")
    pet_reg.plot(ax=ax_inset, color="red", label="PET")
    ax_inset.fill_between(x=pet_reg.month, y1=pet_reg-pet_reg_std, y2=pet_reg+pet_reg_std, color="red", alpha=0.2)
    ax_inset.set_title(title, color="black")
    ax_inset.set_ylabel("[mm/month]", fontsize=8)
    ax_inset.set_xlabel(" ")

    
    # Calculate total annual precipitation and PET
    total_pr_ann = pr_reg.sum().values
    total_pet_ann = pet_reg.sum().values
    ax_inset.text(12.5, 12+165, f"PET:{total_pet_ann:.0f} mm/yr", color="red", fontsize=8, horizontalalignment="right")
    ax_inset.text(12.5, 10.5+145, f"PR:{total_pr_ann:.0f} mm/yr", color="tab:blue", fontsize=8, horizontalalignment="right")

    #Add squares around regions
    #WEU
    ax.add_patch(mpatches.Rectangle(xy=[-3, 45], width=17, height=12,
                                facecolor='none', edgecolor='black', alpha=0.5, linestyle="--",
                                transform=ccrs.PlateCarree()))
    if region_name=="WEU":
        ax.annotate("", xy=(inset_center_x+60, inset_center_y-35), xytext=(box_center_lon+43, box_center_lat-4),
                        arrowprops=dict(edgecolor='black', linestyle="--", arrowstyle='-', alpha=0.5, linewidth=1.5), fontsize=12, alpha=0.5, transform=ccrs.PlateCarree())
        ax.annotate("", xy=(inset_center_x+92, inset_center_y+25), xytext=(box_center_lon+43, box_center_lat+7),
                        arrowprops=dict(edgecolor='black', linestyle="--", arrowstyle='-', alpha=0.5, linewidth=1.5), fontsize=12, alpha=0.5, transform=ccrs.PlateCarree())
        
    #SA
    ax.add_patch(mpatches.Rectangle(xy=[14, -35], width=19, height=15,
                                facecolor='none', edgecolor='black', alpha=0.5, linestyle="--",
                                transform=ccrs.PlateCarree())) 
    if region_name=="SA":
        ax.annotate("", xy=(inset_center_x+54, inset_center_y-18.5), xytext=(box_center_lon+59, box_center_lat-8),
                        arrowprops=dict(edgecolor='black', linestyle="--", arrowstyle='-', alpha=0.5, linewidth=1.5), fontsize=12, alpha=0.5, transform=ccrs.PlateCarree())
        ax.annotate("", xy=(inset_center_x+44, inset_center_y+38.5), xytext=(box_center_lon+59.5, box_center_lat+6.5),
                        arrowprops=dict(edgecolor='black', linestyle="--", arrowstyle='-', alpha=0.5, linewidth=1.5), fontsize=12, alpha=0.5, transform=ccrs.PlateCarree())
        
    #Add anothe inset ax on the other side of the main figure
    #ax_inset2 = fig.add_axes([left+0.375, bottom-0.23, width+0.05, height+0.5], projection=ccrs.EqualEarth())
    if region_name=="WEU":
        ax_inset2 = fig.add_axes([left-0.375, bottom-0.24, width+0.05, height+0.5], projection=ccrs.EqualEarth())
        ax_inset2.set_extent([-3, 14, 45, 57], crs=ccrs.PlateCarree())
        ax_inset2.stock_img()
        ax_inset2.coastlines(linewidth=0.5)
        ax_inset2.contour(mask_WEU.lon, mask_WEU.lat, np.isnan(mask_WEU), colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
        ax_inset2.add_feature(cfeature.LAKES)
        ax_inset2.add_feature(rivers_10m, edgecolor="dodgerblue", linewidth=0.8)
        ax_inset2.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="dashed")
        #ax_inset2.set_title("Western Europe (rivers)", fontsize=8)
        ax_inset2.annotate("", xy=(inset_center_x+10, inset_center_y), xytext=(box_center_lon+46, box_center_lat),
                        arrowprops=dict(facecolor='black', arrowstyle='<-', linewidth=2), fontsize=12, transform=ccrs.PlateCarree())

    elif region_name=="SA":
        ax_inset2 = fig.add_axes([left-0.375, bottom-0.23, width+0.05, height+0.5], projection=ccrs.EqualEarth())
        ax_inset2.set_extent([14, 33, -35, -20], crs=ccrs.PlateCarree())
        ax_inset2.stock_img()
        ax_inset2.coastlines(linewidth=0.5)
        ax_inset2.contour(mask_SA.lon, mask_SA.lat, np.isnan(mask_SA), colors='black', linewidths=1.5, transform=ccrs.PlateCarree())
        ax_inset2.add_feature(cfeature.LAKES)
        ax_inset2.add_feature(rivers_10m, edgecolor="dodgerblue", linewidth=0.8)
        ax_inset2.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="dashed")
        #ax_inset2.set_title("South Africa (rivers)", fontsize=8)
        ax_inset2.annotate("", xy=(inset_center_x+12, inset_center_y), xytext=(box_center_lon+62.5, box_center_lat),
                        arrowprops=dict(facecolor='black', arrowstyle='<-', linewidth=2), fontsize=12, transform=ccrs.PlateCarree())
        
filename = "Figure_1"
fig.savefig(f"{filename}.jpg", dpi=1200, bbox_inches="tight")
fig.savefig(f"{filename}.pdf", bbox_inches="tight")