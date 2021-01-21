import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.array
import cartopy.crs as ccrs
import matplotlib.colors as colors
import datetime as dt
from matplotlib.colors import BoundaryNorm
import sys
import warnings
import glob
warnings.filterwarnings('ignore')

import matplotlib.gridspec as gridspec




'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~Loiding in Files~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''



def load_awap():
    directory  = '/g/data/w40/ab2313/'
    path = directory + 'precip_calib_0.25_1911_2017_land.nc'
    AWAP = xr.open_dataset(path)
    
    # Applying the land sea mask
    path = directory  + 'precip_calib_0.25_maskforCAus.nc'
    mask = xr.open_dataset(path)
    mask = mask.rename({'longitude':'lon'})
    mask = mask.rename({'latitude':'lat'})

    AWAP = AWAP.where(mask.mask == 1)

    
     # Just the north
    AWAP = AWAP.sel(lat = slice(-23,0))
    #Rainday > 1mm
    AWAP = AWAP.where(AWAP.precip >= 1, drop = True) 
     # This are unphysical
    AWAP = AWAP.where(AWAP.precip < 8000, drop = True)
    # wet season only
    AWAP = AWAP.where(AWAP.time.dt.month.isin([10,11,12,1,2,3]) , drop = True) 
    
    AWAP['time'] = AWAP.time.values - pd.to_timedelta('9h')
    
    AWAP.attrs = {'Information':'Only contains the wet season [10,11,12,1,2,3],'
               + 'rainfall >= 1mm and the North of Australia'}
    
    return AWAP




    
def load_rmm():
    
    import urllib
    import io

    url = 'http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt'
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0'
    headers={'User-Agent':user_agent}
    request=urllib.request.Request(url,None,headers)
    response = urllib.request.urlopen(request)
    data = response.read()
    csv = io.StringIO(data.decode('utf-8'))

    rmm_df = pd.read_csv(csv, sep=r'\s+', header=None, skiprows=2,
        usecols=[0,1,2,3,4,5,6,7], names=['year', 'month', 'day','RMM1','RMM2', 'phase', 'amplitude', 'origin'])
    index = pd.to_datetime(rmm_df.loc[:,['year','month','day']])
    rmm_df.index = index

    rmm_xr = rmm_df.loc[:,['RMM1','RMM2', 'phase','amplitude']].to_xarray().rename({'index':'time'})
    
    return rmm_xr
    
    
    
def load_access(access_directory, chunks = 0):
    
    access_files = glob.glob(access_directory + '*.nc')
    
    
    '''Combining all of the ensemble files into the 1 xarray file'''

    ensemble_stor  = []

    for ensemble_path in access_files:
        if chunks:
             single_ensemble = xr.open_dataset(ensemble_path, chunks = {'time':-1}).sel(latitude  = slice(-23,-10))
        else:
            single_ensemble = xr.open_dataset(ensemble_path).sel(latitude  = slice(-23,-10))
        ensemble_stor.append(single_ensemble)

    access = xr.concat(ensemble_stor, pd.Index(np.arange(1, len(ensemble_stor) + 1), name = 'ensemble'))
   
    # Making the time start at 00 rather than 12
    access['time'] = access.time.values - pd.to_timedelta('12h')
    
    # Only want raindays
    access = access.where(access.pr >= 1, drop = True) 
    
    # Renaming in order to be used with other functions
    access = access.rename({'pr':'precip'})
    access = access.rename({'longitude':'lon', 'latitude':'lat'})
    
    # Attributes
    access.attrs = {'Information':'Only contains the wet season [10,11,12,1,2,3],'
               + 'rainfall >= 1mm and the North of Australia'
                   ,'Time':'Only for the wet-season'}
    
    
    
    return access


def load_rmm_access():
    
    
    '''Reading in and restyling rmm'''
    RMM = xr.open_dataset('/g/data/w40/ab2313/RMM/rmm_access_4.nc', chunks = {'time':-1})
    
    # The rmm has one extra ensemble member, I think number 11 is 
    # the ensemble mean
    rmm = RMM.isel(ensemble = slice(0,11)) 
    
    # Resseting so they match access
    rmm['ensemble'] = np.arange(1,12) 
    
    rmm = rmm.where(rmm.time.dt.month.isin([10,11,12,1,2,3]), drop = True)
    
    return rmm









def load_accessifies_variables():
    
    # Loading in All the Datasets
    rmm_access = load_rmm_access()
    rmm_obs = load_rmm()
    awap = load_awap()
    
    access_directory = '/g/data/w40/ab2313/ACCESS_S_1ST_1M_ensembles/'
    access = load_access(access_directory, chunks = 1)
    
    
    # Time matching all of the dataset
    
    awap = awap.where(awap.time.isin(access.time.values), drop = True)

    access = access.where(access.time.isin(awap.time.values), drop = True)
    
    
    
    rmm_access = rmm_access.where(rmm_access.time.isin(access.time.values), drop = True)
    rmm_obs = rmm_obs.where(rmm_obs.time.isin(access.time.values), drop = True)
    
    print(len(rmm_obs.time.values), len(rmm_access.time.values))
    print(len(access.time.values), len(awap.time.values))
    
    return awap, access, rmm_obs, rmm_access
    
    
def load_accessifies_variables2():
    
    
    datadir = '/g/data/w40/ab2313/accessified_vars'
    # Loading in All the Datasets
    awap = xr.open_dataset(datadir + 'awap.nc')
    access = xr.open_dataset(datadir + 'access.nc')
    rmm_obs = xr.open_dataset(datadir + 'rmm_obs.nc')
    rmm_access = xr.open_dataset(datadir + 'rmm_access.nc')
    
    return awap, access, rmm_obs, rmm_access