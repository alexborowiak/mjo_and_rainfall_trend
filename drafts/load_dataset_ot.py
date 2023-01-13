import xarray as xr
from constants import MJO_DATA_PATH
import os
import numpy as np
import pandas as pd

# +
# def load_awap(fname= 'precip_calib_0.25_1911_2017_land.nc'):
#     awap = xr.open_dataset(os.path.join(MJO_DATA_PATH, fname))
#     return awap


def load_awap(fname= 'precip_calib_0.25_1911_2017_land.nc'):
    awap_path  = os.path.join(MJO_DATA_PATH, fname)
    print(f'Opening awap data from {awap_path}')
    AWAP = xr.open_dataset(awap_path)
    
    # Applying the land sea mask
    mask_path =  os.path.join(MJO_DATA_PATH,'precip_calib_0.25_maskforCAus.nc')
    print(f'Appling mask from {mask_path}')
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


# -

def load_ot_rmm(fname='mjoindex_IHR_20CRV2c.nc'):
    
    rmm_oliver_path = os.path.join(MJO_DATA_PATH, fname)
    print(f'Opening OT RMM from {rmm_oliver_path}')
    rmm_oliver = xr.open_dataset(rmm_oliver_path)
    rmm_oliver['time'] = pd.date_range(start = '1905-01-01', periods = len(rmm_oliver.time), freq='D')
    #xr.cftime_range(start = '1905-01-01', periods = len(rmm_oliver.time), freq='D')
    rmm_oliver = rmm_oliver.rename({'IHR_amp':'amplitude', 'IHR_phase': 'phase', 'IHR1': 'RMM1', 'IHR2': 'RMM2'})
    return rmm_oliver

       
