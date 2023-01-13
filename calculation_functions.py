import xarray as xr

def match_dataset_time(d1, d2):
    # Making sure the two are the same length
    d1 = d1.where(d1.time.isin(d2.time.values), drop = True)
    d2 = d2.where(d2.time.isin(d1.time.values), drop = True)
    print(f'New datasets are now of lenght d1 = {len(d1.time.values)} and d2 = {len(d1.time.values)}')
    return d1, d2

def convert_time_to_year(ds):
    # The index needs to be changed to just the year. Other wise it will be the full date.
    ds['time'] = ds.time.dt.year
    ds = ds.rename({'time':'year'})
    
    return ds
    

def resample_sum(awap: xr.Dataset):
    awap_resampled = awap.resample(time = 'y').sum(dim = 'time')
    
    awap_resampled = convert_time_to_year(awap_resampled)

    return awap_resampled



def resample_count(awap: xr.Dataset):
    # Resampling: getting the number of raindays each year (each year is a wet season).
    awap_count = awap.resample(time = 'y').count(dim = 'time')
    
    awap_count = convert_time_to_year(awap_count)

    
    return awap_count