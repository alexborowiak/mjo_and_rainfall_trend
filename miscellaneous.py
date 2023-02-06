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

MJO_DIR = '/g/data/w40/ab2313/mjo_and_rainfall_trend/'



def remove_outside_point(data, vmax, vmin):
    '''
    This is for removing the points above and below the threshold. This is specifically 
    for the comparison or anomaly data set types 
    '''
    
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



def apply_masks(data):
    # The land sea maks for era-5
    directory = MJO_DIR
    landsea_mask = xr.open_dataset(directory + 'mask_landsea.nc')
    landsea_mask = landsea_mask.rename({'longitude':'lon', 'latitude':'lat'})
    landsea_mask = landsea_mask.squeeze().drop('time')
    data = data.where(landsea_mask.lsm >= 0.5, drop = True)
    
    
    # the andrew mask for the gibson desert
    path = directory  + 'precip_calib_0.25_maskforCAus.nc'
    mask = xr.open_dataset(path)
    mask = mask.rename({'longitude':'lon'})
    mask = mask.rename({'latitude':'lat'})
    
    data = data.where(mask.mask == 1)
    return data



