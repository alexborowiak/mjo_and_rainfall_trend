import xarray as xr
import numpy as np
import dask.array
import warnings
import pandas as pd
import os
warnings.filterwarnings('ignore')
from constants import MJO_DATA_PATH



def load_mask():
    # the andrew mask for the gibson desert
    path = os.path.join(MJO_DATA_PATH, 'precip_calib_0.25_maskforCAus.nc')
    mask = xr.open_dataset(path)
    mask = mask.rename({'longitude':'lon'})
    mask = mask.rename({'latitude':'lat'})
    
    return mask



def load_awap():
    awap_path =  os.path.join(MJO_DATA_PATH, 'awap_full.nc')
    AWAP = xr.open_dataset(awap_path)
    
    # Applying the land sea mask
    mask_path =  os.path.join(MJO_DATA_PATH, 'precip_calib_0.25_maskforCAus.nc')
    mask = xr.open_dataset(mask_path)
    mask = mask.rename({'longitude':'lon'})
    mask = mask.rename({'latitude':'lat'})

    AWAP = AWAP.where(mask.mask == 1)

    #Rainday > 1mm
    AWAP = AWAP.where(AWAP.precip >= 1, drop = True) 
     # This are unphysical
    AWAP = AWAP.where(AWAP.precip < 8000, drop = True)
    
    # Rainday starets at 9am, we want day to be a day
    AWAP['time'] = AWAP.time.values - pd.to_timedelta('9h')
    
    AWAP.attrs = {'Information':'Only contains the wet season [10,11,12,1,2,3],'
               + 'rainfall >= 1mm and the North of Australia',
                 'History': 'AGCD Regrid on the 11th of June 2021 from /g/data/rr5 by ab2313'}
    
    return AWAP


   

def load_rmm():
    
    import urllib
    import io

    url = 'http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt'
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0'
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

    
def load_ot_rmm(fname='mjoindex_IHR_20CRV2c.nc'):
    
    rmm_oliver_path = os.path.join(MJO_DATA_PATH, fname)
    print(f'Opening OT RMM from {rmm_oliver_path}')
    rmm_oliver = xr.open_dataset(rmm_oliver_path)
    rmm_oliver['time'] = pd.date_range(start = '1905-01-01', periods = len(rmm_oliver.time), freq='D')
    #xr.cftime_range(start = '1905-01-01', periods = len(rmm_oliver.time), freq='D')
    rmm_oliver = rmm_oliver.rename({'IHR_amp':'amplitude', 'IHR_phase': 'phase', 'IHR1': 'RMM1', 'IHR2': 'RMM2'})
    return rmm_oliver
