"""
This script computes domain-integrated heat content sea-ice and other tracer diagnostics using 1/4 deg NEMO simulations of the North Atlantic Ocean (run at University of Alberta).
This is setup for ANHA4-EPM111 simulation. See details on the link below.
https://canadian-nemo-ocean-modelling-forum-commuity-of-practice.readthedocs.io/en/latest/Institutions/UofA/Configurations/ANHA4/index.html#

Currently, the script is set up for computing heat content and sea-ice area subpolar North Atlantic and Arctic oceans.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import glob
import os

import warnings
warnings.filterwarnings("ignore")


### ------ Functions for computations ----------
def area_sum(ds, dA = 1., x='X', y='Y'):
    """Compute spatial-sums
    Parameters
    ----------
    ds : xarray Dataset for data variables
    dA : xarray Dataset for cell areas
    
    Returns
    -------
    ds_sum : timeseris of spatially-integrated dataset
    """
    
    ds_sum = (ds * dA).sum([x, y])
    
    return ds_sum


### ------ Main calculations ------------------

ppdir = "/mnt/storage6/myers/NEMO/ANHA4-EPM111/"

ppdir_grid = "/mnt/storage6/myers/NEMO_meshes/ANHA4/50VerticalLevels/"

ppdir_save = "/mnt/storage6/hemant/Memory-ANHA4-EPM111/timeseries/"

# read grid data
ds_grid = xr.open_dataset(ppdir_grid + "mesh_mask_anha4.nc")

for year in range(1959, 2022):

    print("Running year = ", year)

    ## ----- read tracer grid data and sea ice data -----------
    ds = xr.open_mfdataset(ppdir + "*" + str(year) + "*gridT.nc")

    ds_ice = xr.open_mfdataset(ppdir + "*" + str(year) + "*icemod.nc")

    ds = xr.merge([ds.rename({'y_grid_T':'y', 'x_grid_T':'x', 'deptht':'z', 'time_counter':'time', 
                         'nav_lat_grid_T':'lat', 'nav_lon_grid_T':'lon'}),
                   ds_ice.rename({'time_counter':'time'}).drop(['nav_lat', 'nav_lon']),
                   ds_grid.get(['e1t', 'e2t', 'tmask', 'nav_lat', 'nav_lon'])])
    
    ds = ds.isel(y=slice(280,len(ds['y']))) # remove latitudes (south of 45N) are not needed

    print("Data read successfully")
    
    ## ----- Ocean heat content calculations (domain integration) -----------
    ds_save = xr.Dataset()
    
    cell_vol = ds['e1t'] * ds['e2t'] * ds['e3t']
    cell_vol = cell_vol.where(ds['tmask'] > 0)

    var = 'votemper'
    # subpolar North Atlantic
    dV = cell_vol.where((ds['nav_lat']>=45.) & (ds['nav_lat']<=70.))
    ds_save[var + '_subpolar_NAtl'] = area_sum(ds['votemper'], dA = dV, x='x', y='y')
    ds_save[var + '_subpolar_NAtl'].attrs['long_name'] = "Subpolar North Atlantic (45N-70N): volume-integrated"
    ds_save['vol_subpolar_NAtl'] = dV.sum(['x', 'y'])

    # Arctic
    dV = cell_vol.where((ds['nav_lat']>=60.))
    ds_save[var + '_arctic'] = area_sum(ds['votemper'], dA = dV, x='x', y='y')
    ds_save[var + '_arctic'].attrs['long_name'] = "Arctic (60N-90N): volume-integrated"
    ds_save['vol_arctic'] = dV.sum(['x', 'y'])

    # subtropical North Atlantic
    #dV = cell_vol.where((ds['nav_lat']>=0.) & (ds['nav_lat']<=45.))
    #ds_save[var + '_subtropical_NAtl'] = area_sum(ds['votemper'], dA = dV, x='x', y='y')
    #ds_save[var + '_subtropical_NAtl'].attrs['long_name'] = "Subtropical North Atlantic (0N-45N): volume-integrated"
    #ds_save['vol_subtropical_NAtl'] = dV.sum(['x', 'y'])

    # North Atlantic
    #dV = cell_vol.where((ds['nav_lat']>=0.) & (ds['nav_lat']<=70.))
    #ds_save[var + '_NAtl'] = area_sum(ds['votemper'], dA = dV, x='x', y='y')
    #ds_save[var + '_NAtl'].attrs['long_name'] = "North Atlantic (0N-70N): volume-integrated"
    #ds_save['vol_NAtl'] = dV.sum(['x', 'y'])
    
    ## ----- Sea ice and heat flux calculations (domain integration) ---------
    cell_area = ds['e2t'] * ds['e1t']
    cell_area = cell_area.where(ds['tmask'].isel(z=0) > 0)

    ds['ice_vol'] = ds['ileadfra'] * ds['iicethic']
    var_list = ['ice_vol', 'ileadfra', 'sohefldo']
    
    for var in var_list:
        if(var == 'sohefldo'):
            # subpolar North Atlantic
            dA = cell_area.where((ds['nav_lat']>=45.) & (ds['nav_lat']<=70.))
            ds_save[var + '_subpolar_NAtl'] = area_sum(ds[var], dA = dA, x='x', y='y')
            ds_save[var + '_subpolar_NAtl'].attrs['long_name'] = "Subpolar North Atlantic (45N-70N): area-integrated"
        
            # Arctic
            dA = cell_area.where((ds['nav_lat']>=60.))
            ds_save[var + '_arctic'] = area_sum(ds[var], dA = dA, x='x', y='y')
            ds_save[var + '_arctic'].attrs['long_name'] = "Arctic (60N-90N): area-integrated"
        
            # subtropical North Atlantic
            #dA = cell_area.where((ds['nav_lat']>=0.) & (ds['nav_lat']<=45.))
            #ds_save[var + '_subtropical_NAtl'] = area_sum(ds[var], dA = dA, x='x', y='y')
            #ds_save[var + '_subtropical_NAtl'].attrs['long_name'] = "Subtropical North Atlantic (0N-45N): area-integrated"
        
            # North Atlantic
            #dA = cell_area.where((ds['nav_lat']>=0.) & (ds['nav_lat']<=70.))
            #ds_save[var + '_NAtl'] = area_sum(ds[var], dA = dA, x='x', y='y')
            #ds_save[var + '_NAtl'].attrs['long_name'] = "North Atlantic (0N-70N): area-integrated"
        
        else:
            ds_save[var] = area_sum(ds[var], dA = cell_area, x='x', y='y')
            ds_save[var].attrs['long_name'] = "Area-integrated"
            
    ds_save['cell_area'] = cell_area
    ds_save = xr.merge([ds_save, ds.get(['nav_lat', 'nav_lon'])])
    
    ## --- Save data -------
    directory = ppdir_save
    if not os.path.exists(directory):
        # If it doesn't exist, create it
        os.makedirs(directory)
                    
    save_file_path = (ppdir_save + "Ocean_temp_ice_" + str(year) + ".nc")
    ds_save = ds_save.astype(np.float32).compute()
    ds_save.to_netcdf(save_file_path)
    
    print("Data saved succefully")




