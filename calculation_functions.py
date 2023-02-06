import xarray as xr
from miscellaneous import apply_masks
import numpy as np


START_MONSOON_MONTHS = [12]
END_MONSOON_MONTHS = [1,2,3]

def match_dataset_time(d1, d2):
    '''
    Matches the time in two xarray dataset or dataarray.
    '''
    d1 = d1.where(d1.time.isin(d2.time.values), drop = True)
    d2 = d2.where(d2.time.isin(d1.time.values), drop = True)
    print(f'New datasets are now of lenght d1 = {len(d1.time.values)} and d2 = {len(d1.time.values)}')
    
    return d1, d2

def convert_time_to_year(ds):
    # The index needs to be changed to just the year. Other wise it will be the full date.
    if 'year' in list(ds.coords):
        print('Year is already the coord - no need to override')
        return ds
    ds['time'] = ds.time.dt.year
    ds = ds.rename({'time':'year'})
    
    return ds    


def _combine_start_and_end_wet_season(ds_start, ds_end):
    ds_end['year'] = ds_end.year.values - 1
    
    ds = ds_start + ds_end
    
    return ds

   
def _resample_sum_ds(ds):
    ds_resample = ds.resample(time = 'y').sum(dim = 'time')
    ds_resample = convert_time_to_year(ds_resample)
    return ds_resample

def _resample_count_ds(ds):
    ds_resample = ds.resample(time = 'y').count(dim = 'time')
    ds_resample= convert_time_to_year(ds_resample)
    return ds_resample


def monsoon_resample(ds: xr.Dataset, method:str):
    ds_start = ds.where(ds.time.dt.month.isin([START_MONSOON_MONTHS]), drop=True)
    ds_end = ds.where(ds.time.dt.month.isin([END_MONSOON_MONTHS]), drop=True)
    
    if method == 'sum':
        func1d = _resample_sum_ds
    elif method == 'count':
        func1d = _resample_count_ds
    # TODO: Add in enum and exceptions. OR convert to method.
        
    ds_start_resample = func1d(ds_start)
    ds_end_resample = func1d(ds_start)
    
    ds = _combine_start_and_end_wet_season(ds_start_resample, ds_end_resample)
    
    return ds
    

def resample_sum(awap: xr.Dataset):
    awap_resampled = awap.resample(time = 'y').sum(dim = 'time')
    awap_resampled = convert_time_to_year(awap_resampled)

    return awap_resampled

def resample_count(awap: xr.Dataset):
    awap_count = awap.resample(time = 'y').count(dim = 'time')
    awap_count = convert_time_to_year(awap_count)
    
    return awap_count


def max_filter(da: xr.DataArray, vmax: float) -> xr.DataArray:
    '''
    Removes values in an xarray data array above vmax and below negative vmax and replaces
    them with vmax and negative vmax respectivly.
    '''
    da = da.where(da < vmax, vmax - 0.01)
    da = da.where(da > -vmax, -vmax + 0.01)
    return da

def moving_average(data:np.ndarray, number_points:int=3):
    '''
    Calculates the moving avaregae of a numpy array.
    '''
    to_return = np.cumsum(data, dtype=float)
    to_return[number_points:] = to_return[number_points:] - to_return[:-number_points]
    return to_return/number_points


@xr.register_dataarray_accessor('calc')
class CalculationMethods:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    def weighted_space_mean(self):
        da = self._obj
        weights = np.cos(np.deg2rad(da.lat))
        weights.name='weights'
        
        da_weighted_mean = da.weighted(weights).mean(dim=['lat', 'lon'])
        return da_weighted_mean
     

