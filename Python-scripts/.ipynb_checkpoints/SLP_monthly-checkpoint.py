""" 
This script reads SLP data from CORE2 and NCEP datasets, and saves the monthly-avearged fields.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import glob
import os

import warnings
warnings.filterwarnings("ignore")

### ------ Main calculations ------------------

ppdir = "/mnt/storage7/myers/DATA/"

ppdir_save = "/mnt/storage6/hemant/Memory-ANHA4-EPM111/timeseries/"


# Compute monthly CORE2 SLP data
for year in range(1948, 2010):
    
    print("CORE2: Running year = ", year)
    
    ds = xr.open_dataset(ppdir + "CORE2-IA/slp_core2_y" + str(year) + ".nc")
    
    ds_month = ds.resample(TIME='1M').mean()
    ds_month = ds_month.rename({'LAT':'lat', 'LON':'lon', 'TIME':'time'})
    
    save_file_path = (ppdir_save + "SLP_CORE2_" + str(year) + ".nc")
    ds_month.to_netcdf(save_file_path)


# Compute monthly NCEP SLP data
for year in range(2001, 2025):
    
    print("NCEP: Running year = ", year)
    
    ds = xr.open_dataset(ppdir + "NCEP_R2/mslp_y" + str(year) + ".nc")
    
    ds_month = ds.resample(time='1M').mean()
    ds_month = ds_month.rename({'pres':'slp'})
    
    save_file_path = (ppdir_save + "SLP_NCEP_" + str(year) + ".nc")
    ds_month.to_netcdf(save_file_path)
