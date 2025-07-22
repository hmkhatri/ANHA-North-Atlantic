"""
This script uses an analytical form of response function (heat content and sea ice change), 
to predict the actual ocean heat content and sea-ice timeseries.

Correlation coefficient between predicted and actual signals can be compuetd for a range of tuning paramters (gamma, beta in response functions) for choosing an optimal set of parameters.
See details of the method in Khatri et al. (2024)
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import glob
import os

import warnings
warnings.filterwarnings("ignore")

# ------- Functions for calculations ----------
def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, dims, deg=1):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, dim, deg=deg)
    return da_detrended

# ------ Main code -----------------

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/HighResMIP/NEMO_UAlberta/Memory-ANHA4-EPM111/"

year_int = 20. # length of response function in years

gamma_rng = np.arange(0.5, 20.1, 0.5)
beta_rng = np.arange(0.0, 10.1, 0.5)

# read data
ds = xr.open_dataset(ppdir + "timeseries/Index_AO_NAO.nc")

# remove linear trends
var_list = ['ao_pcs', 'nao_pcs']
for var in var_list:
    ds[var] = detrend(ds[var], ['time'])
ds = ds.drop('month')

# Create impulse response functions 
tim = (ds['time'].dt.year + ds['time'].dt.month / 12 
    - ds['time.year'].values[0] - 10.) # temporary timeseries
fac = 12 # for monthly data

Response_function_full = []
for gamma in gamma_rng:
    
    # loop for testing beta for sinusodial damping 
    Response_function_gamma = []
            
    for beta in beta_rng:
    
        # Response function for Sinusoidal Damping term
        Response_function1 = np.exp(-(tim) / (gamma)) * ( np.cos(tim * beta / gamma))
        Response_function1 = xr.where(tim < 0., 0., Response_function1)
        Response_function1 = xr.where((tim > 1.5 * np.pi * gamma/beta), 0., Response_function1)
            
        Response_function1 = Response_function1.isel(time=slice(10*fac - 1, 10*fac-1 +
                                                                int(fac*year_int)))
        Response_function1 = Response_function1.isel(time=slice(None, 
                                                                None, -1)).drop(['time'])
    
        Response_function_gamma.append(Response_function1)
    
    Response_function_gamma = xr.concat(Response_function_gamma, dim="beta")
    
    Response_function_full.append(Response_function_gamma)
    
Response_function_full = xr.concat(Response_function_full, dim="gamma")

# Predicttions using response functions and heat flux timseries ------
Pred_AO = xr.zeros_like(ds['ao_pcs'])
Pred_AO = Pred_AO / Pred_AO

(tmp, Pred_AO) = xr.broadcast(Response_function_full.isel(time=0), Pred_AO)
Pred_AO = Pred_AO.copy() # otherwise runs into "assignment destination is read-only" error

Pred_NAO = Pred_AO.copy()

for j in range(0 + int(fac*year_int), len(ds['time'])):
                    
    tmp1 = ds['ao_pcs'].isel(time=slice(j-int(fac*year_int), j))
    Pred_AO[:,:,:,j] = (tmp1 * Response_function_full).sum('time')
    
    tmp1 = ds['nao_pcs'].isel(time=slice(j-int(fac*year_int), j))
    Pred_NAO[:,:,:,j] = (tmp1 * Response_function_full).sum('time')

print("Impulse response calculations completed")

ds_save = ds.copy()

ds_save['Pred_AO'] = Pred_AO
ds_save['Pred_AO'].attrs['long_name'] = ("Reconstructions using SLP EOF timeseries in" + 
                                        " the Arctic (60N-90N)")
ds_save['Pred_NAO'] = Pred_NAO
ds_save['Pred_NAO'].attrs['long_name'] = ("Reconstructions using SLP EOF (20N-80N, 90W-40E)" + 
                                          " timeseries in the subpolar North Atlantic")

ds_save = ds_save.assign_coords(gamma = gamma_rng)
ds_save = ds_save.assign_coords(beta = beta_rng)

save_file_path = ppdir + "Recons_green_function/Response_SLP_EOF.nc" 

ds_save = ds_save.astype(np.float32).compute()
ds_save.to_netcdf(save_file_path)
    
print("Data saved succefully")