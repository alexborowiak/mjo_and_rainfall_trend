import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.array
import cartopy.crs as ccrs
import matplotlib.colors as colors
import datetime as dt
from matplotlib.colors import BoundaryNorm
import matplotlib.gridspec as gridspec
import sys
import warnings
import glob
warnings.filterwarnings('ignore')



# This is for removing the points above and below the threshold. This is specifically 
# for the comparison or anomaly data set types 
def remove_outside_point(data, vmax, vmin):
        
        
        # The points need to be slightly within the range
        vmax = vmax - 0.0001
        vmin = vmin + 0.0001
        
        # Filling the nans with 1. 1 is chosen as it is the default value (means equal to climatology) in this case.
        # This is imortant as nan is considered both above and below vmin and vmax. Gets sticky otherwise
        data = data.fillna(1)
        # The points that are greater than vmin are true, and the other (< vmin) are replaced with vmin
        data = data.where(data > vmin, vmin)
        # Samne as above except for vmax
        data = data.where(data < vmax, vmax)
        # WHere the data is equal to 1, replace with nan. (Think this is broken but here as place holder anyway)
        data = data.where(data != 1, np.nan)
        
        
        return data
    
      
def upper_low_bound(vmin, vmax):
    
    if vmin == 0:
        lower_bound = 0
    else:
        low_mag = OrderOfMagnitude(vmin) * 10
        lower_bound = np.floor(vmin/low_mag) * low_mag
        
        
    if vmax == 0:
        upper_bound = 0
    else:
        
        high_mag = OrderOfMagnitude(vmax) * 10
        upper_bound = np.ceil(vmax/high_mag) * high_mag
    
    
    return lower_bound, upper_bound
    

    
def OrderOfMagnitude(number):
    import math
    mag = math.floor(math.log(number, 10))   
    if mag == 0:
        return 0.1
    else:
        return mag
     
                
def apply_masks(data):
    # The land sea maks for era-5
    import xarray as xr
    directory2 = '/g/data/w40/ab2313/ERA5/'
    landsea_mask = xr.open_dataset(directory2 + 'mask_landsea.nc')
    landsea_mask = landsea_mask.rename({'longitude':'lon', 'latitude':'lat'})
    landsea_mask = landsea_mask.squeeze().drop('time')
    data = data.where(landsea_mask.lsm >= 0.5, drop = True)
    
    
    # the andrew mask for the gibson desert
    directory  = '/g/data/w40/ab2313/'
    path = directory  + 'precip_calib_0.25_maskforCAus.nc'
    mask = xr.open_dataset(path)
    mask = mask.rename({'longitude':'lon'})
    mask = mask.rename({'latitude':'lat'})
    
    data = data.where(mask.mask == 1)
    return data



# This is a special mask for ranks. This replaces the masked points with 999, rather than np.nan. This helps with calculation
# speed
def apply_masks_ranks(data):
    # The land sea maks for era-5
    import xarray as xr
    directory2 = '/g/data/w40/ab2313/ERA5/'
    landsea_mask = xr.open_dataset(directory2 + 'mask_landsea.nc')
    landsea_mask = landsea_mask.rename({'longitude':'lon', 'latitude':'lat'})
    landsea_mask = landsea_mask.squeeze().drop('time')
    landsea_mask['lat'] = landsea_mask.lat.values.astype(float)
    landsea_mask['lon'] = landsea_mask.lon.values.astype(float)
    landsea_mask = landsea_mask.where(landsea_mask.lat.isin(data.lat.values) & landsea_mask.lon.isin(data.lon.values),drop = True)
    data = data.sortby('lat', ascending = False)

    data = data.where(landsea_mask.lsm >= 0.6)
    
    
    # the andrew mask for the gibson desert
    directory  = '/g/data/w40/ab2313/'
    path = directory  + 'precip_calib_0.25_maskforCAus.nc'
    mask = xr.open_dataset(path)
    mask = mask.rename({'longitude':'lon'})
    mask = mask.rename({'latitude':'lat'})
    mask['lat'] = mask.lat.values.astype(float)
    mask['lon'] = mask.lon.values.astype(float)
    mask = mask.where(mask.lat.isin(data.lat.values) & mask.lon.isin(data.lon.values),drop = True)
    mask = mask.sortby('lat', ascending = False)
    data = data.where(mask.mask == 1)
    return data