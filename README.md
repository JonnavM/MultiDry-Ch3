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
	2_calculate_SVD.py: Calculates the SVD patterns and timeseries
	2_create_areacello.py: Creates a file for weighted grid calculations
	2_climate_indices.py: Calculates timeseries of known climate indices
	2_std_anom.py: Calculates standardised anomalies of timeseries
	
	3. Scripts for figures
	3_Regression.py: Figures 3 (boxplot), 5 (PR and PET anomalies), 6 (linear regressions)
	3_SPEI12-06-03.py: Plots SPEI12, SPEI06, SPEI03, and SPEI01 to show development of MYDs. Results in Figure 4.
	3_SPEI_figures.py: Makes figures of SPEI-12 per region. Results in Figure 2.
	3_autocorrelation_trends.py: Makes plots for the lagged auto-correlation of SPEI-12, precipitation, and PET. Also plots the trends of all variables included in PET. Results in Figures 7 (lagged auto-correlations), and S11-16 (trends)
	3_worldmaps_climate.py: Makes maps of climatology of PR, PET, number of MYDs, duration of MYDs, intensity of MYDs, fraction of months in MYDs compared to NDs. Results in Figures 1 and S1.

b. Masks: 

	Contains masks for central Argentina (ARG), Southeast Australia (AUS), California (CAL), India (IND), South Africa (SA), and Western Europe (WEU). All masks are based on (combinations of) river basins.
