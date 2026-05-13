This repository provides the necessary scripts to reproduce the results and figures from the paper "Contrasting Drivers of Single-Year and Multi-Year Droughts in Western Europe and Southern Africa". 
It contains the following folders and content:

a. Scripts:

	0. Scripts with functions 
	0_functions.py: contains the functions to calculate the multi-year droughts (MYD), the normal droughts (ND), and creates masks for these droughts (mask_MYD and mask_ND). These functions are loaded in in the other scripts where necessary. 
 
	1. Scripts to calculate PET, SPEI and drought masks
	1_detrend_data.py: Detrends different data field wrt the GMST
	1_PenmanMonteith.py: Calculates PET with use of CMIP6 data
	1_calculate_SPEI.py: Calculates SPEI with use of PET and PR
	1_masks_MYD_ND.py: Creates masks for MYDs and NDs over time
	1_PET_PR_concat.py: Concatinates and calculates PR-12 and PET-12 over all models
 
	2. Scripts to calculate the SVD, the SVD-timeseries, and other input files
	2_calculate_SVD.py: Calculates the SVD patterns and timeseries, plots Figure S5
	2_create_areacello.py: Creates a file for weighted grid calculations
	2_climate_indices.py: Calculates timeseries of known climate indices
	2_std_anom.py: Calculates standardised anomalies of timeseries
	
	3. Scripts for figures
	3_plot_regions.py: Figure 1
	3_plot_drivers.py: Figures 3 and 4
	3_lagged_corr.py: Figures 5, 6, S6, S7
	3_strength_drivers.py: Figures 7 and 8
	3_heatmap_MYDs.py: Figure S4

b. Masks: 

	Contains masks for Southern Africa (SA) and Western Europe (WEU). All masks are based on (combinations of) river basins on a 1x1 grid.
