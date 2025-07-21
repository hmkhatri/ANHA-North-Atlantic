"""
This script computes Arctic Oscillation and North Atlantic Oscillation indices using sea level pressure (SLP) data.
SLP data from CORE2 (1958-2009) and NCEP (2001-2021) datasets is used.

First, a single timeseries is created by combining CORE2 (1958-2000) and NCEP (2001-2021) data.
Then climate indices are computed using EOF analysis

1. Arctic Oscillation (AO) - SLP anomalies north of 20N are used
2. North Atlantic Oscillation (NAO) - SLP anomalies at 20N-80N and 90W-40E are used.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
import xeofs as xe

import warnings
warnings.filterwarnings("ignore")

### ------ Main calculations ------------------

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/HighResMIP/NEMO_UAlberta/Memory-ANHA4-EPM111/timeseries/"

# read data and combine them
ds_core2 = xr.open_mfdataset(ppdir + "SLP_CORE*nc")
#ds_core2 = ds_core2.sel(time = ds_core2['time.year'] >= 1958);
#ds_core2 = ds_core2.sel(time = ds_core2['time.year'] <= 2000);

#ds_ncep = xr.open_mfdataset(ppdir + "SLP_NCEP*nc")
#ds_ncep = ds_ncep.sel(time = ds_ncep['time.year'] <= 2021);
#ds_ncep = ds_ncep.convert_calendar("noleap") # to convert to cftime for consistency

#ds = xr.concat([ds_core2.drop(['lat', 'lon']), ds_ncep.get(['slp'])], dim='time') # combine slp data

ds = ds_core2

#### Climate Indices ----

ds_save = xr.Dataset()

# remove climatology and time-mean
ds_clim = ds.groupby('time.month').mean('time') 
ds_anom = ds.groupby('time.month') - ds_clim
ds_anom = ds_anom.compute()

#ds_anom = ds_anom - ds_anom.mean('time') 

# Compute AO index using EOFs over the Northen hemisphere
psl_anom = ds_anom['slp'].where((ds_anom['lat'] >= 60.) & (ds_anom['lat'] <= 90.)) # latitude range 20N-90N

model_eof = xe.single.EOF(n_modes=10, use_coslat=True, compute=False)
model_eof.fit(psl_anom, dim='time')

model_eof.compute()

ds_save['ao_expvar'] = model_eof.explained_variance_ratio()
ds_save['ao_expvar'].attrs['long_name'] = "EOF-PCA: Explained variance"
            
ds_save['ao_eofs'] = model_eof.components()
ds_save['ao_eofs'].attrs['long_name'] = "EOF-PCA: EOF modes"
            
ds_save['ao_pcs'] = model_eof.scores()
ds_save['ao_pcs'].attrs['long_name'] = "EOF-PCA: Principal Components"

del ds_save['ao_expvar'].attrs['solver_kwargs']
del ds_save['ao_eofs'].attrs['solver_kwargs']
del ds_save['ao_pcs'].attrs['solver_kwargs']

# Compute NAO index using EOFs over the North Atlantic
psl_anom = ds_anom['slp'].where((ds_anom['lat'] >= 20.) & (ds_anom['lat'] <= 80.)) # latitude range 20N-80N
psl_anom = psl_anom.where((psl_anom['lon'] <= 40.) | (psl_anom['lon'] >= 270.)) #longitude range 90W-40E

model_eof = xe.single.EOF(n_modes=10, use_coslat=True, compute=False)
model_eof.fit(psl_anom, dim='time')

model_eof.compute()

ds_save['nao_expvar'] = model_eof.explained_variance_ratio()
ds_save['nao_expvar'].attrs['long_name'] = "EOF-PCA: Explained variance"
            
ds_save['nao_eofs'] = model_eof.components()
ds_save['nao_eofs'].attrs['long_name'] = "EOF-PCA: EOF modes"
            
ds_save['nao_pcs'] = model_eof.scores()
ds_save['nao_pcs'].attrs['long_name'] = "EOF-PCA: Principal Components"

del ds_save['nao_expvar'].attrs['solver_kwargs']
del ds_save['nao_eofs'].attrs['solver_kwargs']
del ds_save['nao_pcs'].attrs['solver_kwargs']

### -------- Save Data --------------
ds_save = ds_save.astype(np.float32).compute()
save_file_path = (ppdir + "Index_AO_NAO.nc")
ds_save.to_netcdf(save_file_path)

