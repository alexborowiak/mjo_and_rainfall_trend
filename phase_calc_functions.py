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
warnings.filterwarnings('ignore')
import mystats
from functools import partial

import matplotlib.gridspec as gridspec
from typing import List

import load_dataset as load

import calculation_functions

enhanced_phases = [5,6]
suppressed_phases = [1,2,3]
transition_phases = [4,7,8]


def wet_season_year(data):
    '''
    This is the start of the wet_season, wet want to move it to the next year so that the start of the
    wet season and the end are both in the one year. This makes it easier for calculatins later on 
    '''
    START_MONSOON_MONTHS = [12]
    END_MONSOON_MONTHS  = [1,2,3]
    
    data_start = data.where(data.time.dt.month.isin(START_MONSOON_MONTHS), drop = True) # The later months of the year
    data_end = data.where(data.time.dt.month.isin(END_MONSOON_MONTHS), drop = True) # The end half
    
    # The end of the monsoon must be moved back to the previous year.
    data_end['time'] = data_end.time - pd.to_timedelta('365day')
    
    total = data_end.combine_first(data_start) # All in one year now :)
    
    return total

def split_into_1to8(datafile, rmm_xr):
    
    rmm_inact_dates = rmm_xr.where(rmm_xr.amplitude < 1, drop = True).time.values
    datafile_inact = datafile.where(datafile.time.isin(rmm_inact_dates), drop = True)


    single_phase = [] # Storage for later concatinating in xarray
    rmm_act = rmm_xr.where(rmm_xr.amplitude >= 1, drop = True) # Only acitve when RMM > 1
    phases = np.arange(1,9) # 8 phases we are looping through
    for phase in phases:
        rmm_single_dates = rmm_act.where(rmm_act.phase == phase, drop = True).time.values # The dates of this phase
        datafile_single = datafile.where(datafile.time.isin(rmm_single_dates), drop = True) # The datafile data in this phase
        single_phase.append(datafile_single) # Appending

    # Update - phase are now integer, with phase 0 being the inactivate phase
    phases = np.append(phases, 0)
    
    # The inactive phase also needs to be included
    single_phase.append(datafile_inact) 

    datafile_RMM_split = xr.concat(single_phase, pd.Index(phases, name = 'phase'))

    return  datafile_RMM_split



def resample_phase_to_subphase(data: xr.DataArray, enhanced_phase_override:List[str]=None) -> xr.Dataset:
    
    used_enhanced_phases = enhanced_phase_override if enhanced_phase_override else enhanced_phases
    print(f'Enhanced phase definition being used {used_enhanced_phases}')
    
    # Legacy: old dataset use string name, whilst new datasets will use phase 0 as inactive phase
    inactive_phase_name = 0 if data.phase.dtype == int else 'inactive'
    
    # Old datatsets will use string, new will use int
    phase_dtype = int if data.phase.dtype == int else str

    enhanced_ds = data.sel(phase = np.array(used_enhanced_phases).astype(phase_dtype)).sum(dim = 'phase')
    suppressed_ds = data.sel(phase = np.array(suppressed_phases).astype(phase_dtype)).sum(dim = 'phase')
    trans_ds = data.sel(phase = np.array(transition_phases).astype(phase_dtype)).sum(dim = 'phase')
    inact_ds = data.sel(phase = inactive_phase_name).drop('phase')
    
    return xr.concat([enhanced_ds,suppressed_ds, trans_ds, inact_ds], 
                     pd.Index(['enhanced','suppressed','transition','inactive'], name = 'phase'))


def count_in_rmm_phase(rmm):
    '''Counts the number of days in each of the MJO phases for each wet-season. This is useful for 
    normalising all of the count trends'''
    
    rmm_act = rmm.where(rmm.amplitude > 1, drop = True)
    phases = np.arange(1,9)
    phase_stor_dict = {}#S[]
    for phase in phases:
        rmm_single_phase = rmm_act.where(rmm_act.phase==phase)
        number_per_year = rmm_single_phase.phase.resample(time='Y').count(dim='time')
        #calculation_functions.monsoon_resample(rmm_single_phase, method='count')
        phase_stor_dict[phase] = number_per_year.values

    # Inactive Phase
    rmm_inact = rmm.where(rmm.amplitude <=1 , drop = True)
    number_per_year_inact = rmm_inact.phase.resample(time='Y').count(dim='time')
    #calculation_functions.monsoon_resample(rmm_inact, method='count')
   
    phase_stor_dict[0] = number_per_year_inact.values
       
    time_dim = 'year' if 'year' in list(rmm_single_phase.coords) else 'time'
    rmm_phase = xr.Dataset({'number':(('phase',time_dim), list(phase_stor_dict.values()))},
                                   {'phase': list(phase_stor_dict.keys()), time_dim: number_per_year[time_dim].values
                                   })

    rmm_phase = calculation_functions.convert_time_to_year(rmm_phase)
    return rmm_phase


def count_in_rmm_subphase(rmm, enhanced_phase_override:List[str]=None):
    used_enhanced_phases = enhanced_phase_override if enhanced_phase_override else enhanced_phases
    print(f'Enhanced phase definition being used {used_enhanced_phases}')

    phase_dict =  {'enhanced': used_enhanced_phases, 'suppressed': suppressed_phases, 'transition': transition_phases}
    
    
    rmm_act = rmm.where(rmm.amplitude > 1, drop = True)
    
    to_combine = {}
    rmm_inact = rmm.where(rmm.amplitude <=1)
    to_combine['inactive'] = rmm_inact.phase.resample(time='Y').count(dim='time')
    #calculation_functions.monsoon_resample(rmm_inact,method='count')
    #

    for phase_name, phase_nums in phase_dict.items():
        rmm_single_phase = rmm_act.where(rmm_act.phase.isin(np.array(phase_nums).astype(float)))
        resample_phase_values = rmm_single_phase.phase.resample(time='Y').count(dim='time')
        #calculation_functions.monsoon_resample(rmm_single_phase, method='count')
        #
                                         
        to_combine[phase_name] = resample_phase_values.values

    rmm_combined_phases = xr.Dataset({'number':(('phase','time'), list(to_combine.values()))},
                                   {'phase':list(to_combine.keys()), 'time': resample_phase_values.time.values})
    
    rmm_combined_phases = calculation_functions.convert_time_to_year(rmm_combined_phases)
    return rmm_combined_phases



def grid_trend(x: np.ndarray,t: np.ndarray):
    '''
    Calculates the trend for each individaul grid cell
    '''
    # If every point is just a nan values. We don't want to do the polyfit calculation. Just return nan
    if np.all(np.isnan(x)):
        return float('nan')
    
    # Getting the gradient of a linear interpolation
    idx = np.isfinite(x) & np.isfinite(t) #checking where the nans are for both
    x = x[idx]
    t = t[idx]
    if len(t) < 5 or len(x) < 5:
        return np.nan
    grad = np.polyfit(t,x,1)[0]
    return grad

def calculate_trend(data: xr.DataArray):
    '''
    Calcualtes the gradient of the trend along the year axis.
    '''
    
    # The axis number that year is
    axis_num = data.get_axis_num('year')
    
    # Applying trends along each grid cell
    percentile_trend_meta = np.apply_along_axis(grid_trend, axis_num, data.values, 
                                                t = data.year.values)
    # Adding back to xarray data array.
    trend = xr.zeros_like(data.isel(year=0).drop('year').squeeze(), dtype=np.float64)
    trend += percentile_trend_meta

    return trend



def convert_to_percent_per_decade(da:xr.DataArray, trend_da:xr.DataArray)->xr.DataArray:
    '''
    Converts a trend from unit/year, to a percetn increase per decade. This is done by first multiplying by
    10 to get the amount the unit changes in 10 years. Next, dividin by the mean value at a grid cell will
    get it as a fraction of the climatology
    '''
    YEARS_IN_DECADE = 10
    # The mean rainfall at a grid_cell
    mean_gridcell_da = da.mean(dim = 'year')
    
    return (trend_da * YEARS_IN_DECADE / mean_gridcell_da) * 100

def calculate_pvals(da: xr.DataArray):
    '''
    Calculates the pvalues using the mann-kendall test.
    This function needs the raw data and not the trend data, as it works off the raw data
    '''
    
    year_axis_num = da.get_axis_num('year')
    
    # Apply the mann_kendall function along the time axis to the da function
    trend_pval_meta = np.apply_along_axis(mystats.mann_kendall, year_axis_num, da)
    
    trend_pval_ds = xr.zeros_like(da.isel(year=0).drop('year')) + trend_pval_meta
    trend_pval_ds = trend_pval_ds.to_dataset(name='pvals')
    
    return trend_pval_ds
    


def significant_trend_calc(data: xr.DataArray, pvals: xr.Dataset) -> xr.DataArray:
    sig_da = data.where(np.logical_and(pvals.pvals >= 0 ,pvals.pvals <= 0.1))

    return sig_da

def values_in_bounds(da:xr.DataArray, pval_da:xr.DataArray=None, lower=0, upper=0.1):   
    ''' This is a newer version of significant_trend_calc that uses two data arrays
    instead of using datasets. This is better as it doesn't required the pval_da
    to have a names data var.'''
    pval_da = da if pval_da is None else pval_da
    return da.where(np.logical_and(pval_da >= lower, pval_da <= upper))


normalise_mapping = {'phase': count_in_rmm_phase, 'subphase': count_in_rmm_subphase}



def normalise_trend(data: xr.Dataset, normalise:str, enhanced_phase_override=None) -> xr.Dataset:
    '''
    Normalises a trend by the number of days in each mjo phase. This can
    be done for either phases or subphases. The normalise method must be set as string
    which then corresponds to a function in a mapping.
    '''
    rmm = load.load_rmm()
    rmm = wet_season_year(rmm)
    
    normalise_func = normalise_mapping[normalise]
    normalise_func = partial(normalise_func, enhanced_phase_override=enhanced_phase_override)\
                                    if enhanced_phase_override is not None else normalise_func
    phase_count = normalise_func(rmm)

    # This should be year here. However, it was changed.
    if 'year' not in list(phase_count.dims):
        print('renaming time to year.')
        phase_count = phase_count.rename({'time':'year'})
        phase_count['year'] = phase_count.year.dt.year.values

    data_yearly = data.where(data.year.isin(phase_count.year.values), drop=True)
    phase_count = phase_count.where(phase_count.year.isin(data.year.values), drop=True)

    # Normlaising by the number of days in each mjo phase or category. 
    data_normalised = (data_yearly/phase_count.number)
    
    return data_normalised
    

def calculate_trend_and_pvalue(da: xr.DataArray, normalise=None, enhanced_phase_override=None):
    '''
    This is a newever version of return_alltrendinfo_custom. This function returns one dataset with 
    multipole data vars instead of instead of multiple datasets.
    '''
    
    if isinstance(normalise, str): da = normalise_trend(da, normalise=normalise, enhanced_phase_override=enhanced_phase_override)

    # Calculates the trend
    trend_da = calculate_trend(da)

    # Convertes to percent per decade
    trend_percent_da = convert_to_percent_per_decade(da, trend_da)

    # Calculates the significant values
    pvals_da =  calculate_pvals(da)

    return xr.merge([trend_da.rename('trend'),
                      trend_percent_da.rename('trend_percent'),
                      pvals_da])
    
    
def return_alltrendinfo_custom(da: xr.DataArray, normalise=None, enhanced_phase_override=None):
    
    if isinstance(normalise, str): da = normalise_trend(da, normalise=normalise, enhanced_phase_override=enhanced_phase_override)

    # Calculates the trend
    trend_da = calculate_trend(da)

    # Convertes to percent per decade
    trend_percent_da = convert_to_percent_per_decade(da, trend_da)

    # Calculates the significant values
    pvals_da =  calculate_pvals(da)

    trend_sig_da = significant_trend_calc(trend_da, pvals_da)
    trend_percent_sig_da = significant_trend_calc(trend_percent_da, pvals_da)

    return trend_da, trend_sig_da, trend_percent_da, trend_percent_sig_da
