#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 11:10:25 2025

@author: Jonna van Mourik
Heatmap for all models for when MYDs are occurring
"""
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

#Load in data
dir = "/your/directory/here/"
models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "EC-Earth3", "MIROC6", "MPI-ESM1-2-LR"]
reg = "WEU"

#SPEI = xr.open_mfdataset(dir+f"SPEI/{model}/1950-2014/SPEI12_monthly_1950_2014_r*_{model}_{reg}.nc", combine="nested", concat_dim="member").__xarray_dataarray_variable__

SPEI_list = []
for model in models:
    spei_model = xr.open_mfdataset(dir+f"SPEI/{model}/new_data/1950-2014/SPEI12_monthly_1950_2014_r*_{model}_{reg}.nc", combine="nested", concat_dim="member").__xarray_dataarray_variable__.expand_dims(model=[model])
    SPEI_list.append(spei_model)
SPEI = xr.concat(SPEI_list, dim="model", coords="minimal")

#Add multi-model mean
SPEI_MMM = SPEI.mean(dim=("member", "model")).expand_dims(model=["MMM"])
SPEI_tot = xr.concat([SPEI, SPEI_MMM], dim="model")

#%% Plot heatmap
spei_df = SPEI_tot.mean("member").to_pandas()
fig = plt.figure() 
ax = sns.heatmap(spei_df, cmap="BrBG", vmin=-1.5, vmax=1.5, cbar_kws={"label": "SPEI-12 [-]"})
ax.set_title(reg)
# Get time index
time_index = spei_df.columns

# Desired years for ticks (every 20 years)
years_to_show = list(range(time_index[0].year, time_index[-1].year + 1, 20))

# Find the first occurrence of each desired year
tick_positions = []
tick_labels = []
seen = set()

for i, t in enumerate(time_index):
    year = t.year
    if year in years_to_show and year not in seen:
        tick_positions.append(i)
        tick_labels.append(str(year))  # <== explicitly convert to string
        seen.add(year)

# Apply ticks and labels
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, fontsize=10)

# Optional: style y-tick labels
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

plt.show()
fig.savefig(f"Figure_S4_{reg}.jpg", dpi=1200, bbox_inches="tight")
fig.savefig(f"Figure_S4_{reg}.pdf", bbox_inches="tight")

