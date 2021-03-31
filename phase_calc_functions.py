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

import matplotlib.gridspec as gridspec



################################### General Functions
################################### General Functions
################################### General Functions
################################### General Functions
################################### General Functions
################################### General Functions
################################### General Functions


# This function moves the start of the wet season [10, 11, 12] to the next year. This means that
# this year is just the data for one wet season

def wet_season_year(data):
    
    # This is the start of the wet_season, wet want to move it to the next year so that the start of the
    # wet season and the end are both in the one year. This makes it easier for calculatins later on 
    
    data_start = data.where(data.time.dt.month.isin([12]), drop = True) # The later months of the year
    data_start['time'] = data_start.time + pd.to_timedelta('365day') # moving them forward a year
    
    data_end = data.where(data.time.dt.month.isin([1,2,3]), drop = True) # The end half
    
    total = data_end.combine_first(data_start) # All in one year now :)
    
    return total


def split_into_1to8(datafile, rmm_xr):
    
    
    
    '''~~~~~~~~~~~~~~~~~~ Inactive Phases'''
    rmm_inact_dates = rmm_xr.where(rmm_xr.amplitude < 1, drop = True).time.values
    datafile_inact = datafile.where(datafile.time.isin(rmm_inact_dates), drop = True)

    '''~~~~~~~~~~~~~~~~~~ Active Phases
    Summary: Looping through all the different RMM phases; getting the dates fro this phase; finding just the rainfall
    in this phase'''
    single_phase = [] # Storage for later concatinating in xarray
    rmm_act = rmm_xr.where(rmm_xr.amplitude >= 1, drop = True) # Only acitve when RMM > 1
    phases = np.arange(1,9) # 8 phases we are looping through
    for phase in phases:
        rmm_single_dates = rmm_act.where(rmm_act.phase == phase, drop = True).time.values # The dates of this phase
        datafile_single = datafile.where(datafile.time.isin(rmm_single_dates), drop = True) # The datafile data in this phase
        single_phase.append(datafile_single) # Appending

    phases = np.append(phases.astype('str'), 'inactive') # The ianctive also needs to be included
    single_phase.append(datafile_inact) 


    # Final File
    datafile_RMM_split = xr.concat(single_phase, pd.Index(phases, name = 'phase'))
    
    
    
    return  datafile_RMM_split



def resample_phase_to_subphase(data):
    
    enhanced = data.sel(phase = ['4','5','6']).sum(dim = 'phase')
    suppressed = data.sel(phase = ['1','2','8']).sum(dim = 'phase')
    trans = data.sel(phase = ['3','7']).sum(dim = 'phase')
    inact = data.sel(phase = 'inactive').drop('phase')
    
    return xr.concat([enhanced,suppressed, trans, inact], 
                     pd.Index(['enhanced','suppressed','transition','inactive'], name = 'phase'))





















############################ MJO Trends
############################ MJO Trends
############################ MJO Trends
############################ MJO Trends



'''Counts the number of days in each of the MJO phases for each wet-season. This is useful for 
normalising all of the count trends'''
def count_in_rmm_phase(rmm):

    rmm_act = rmm.where(rmm.amplitude > 1, drop = True)
    
    phases = np.arange(1,9)
    single_phase = []
    for phase in phases:

         # Just the data for this single rmm phase
        rmm_single_phase = rmm_act.where(rmm_act.phase == phase)
         # Resmapling via year, to get the number of days in each phase
        number_per_year = rmm_single_phase.phase.resample(time = 'y').count(dim = 'time')
        # Appending
        single_phase.append(number_per_year.values)



    '''Inactive Phase'''
    rmm_inact = rmm.where(rmm.amplitude <=1 , drop = True)
    number_per_year_inact = rmm_inact.phase.resample(time = 'y').count(dim = 'time')

    single_phase.append(number_per_year_inact.values)

    titles = np.append(np.array([str(phase) for phase in phases]),['inactive'])
   
    datafile_RMM_split = xr.Dataset({'number':(('phase','year'), single_phase)},
                                   {'phase':titles,
                                    'year': number_per_year.time.dt.year.values
                                   })
    
   
    
    return datafile_RMM_split


def count_in_rmm_subphase(rmm):
    
    enhanced = [4,5,6]
    suppressed = [1,2,8]
    transition = [3,7]

    phase_dict =  {'enhanced': enhanced, 'suppressed': suppressed, 'transition': transition}
    single_phase = []
    
    rmm_act = rmm.where(rmm.amplitude > 1, drop = True)

    for phase_name, phase_nums in phase_dict.items():

         # Just the data for this single rmm phase
        rmm_single_phase = rmm_act.where(rmm_act.phase.isin(phase_nums))#, drop = True)
         # Resmapling via year, to get the number of days in each phase
        number_per_year = rmm_single_phase.phase.resample(time = 'y').count(dim = 'time')
        # Appending
        single_phase.append(number_per_year.values)



    '''Inactive Phase'''
    rmm_inact = rmm.where(rmm.amplitude <=1)# , drop = True)
    number_per_year_inact = rmm_inact.phase.resample(time = 'y').count(dim = 'time')

    single_phase.append(number_per_year_inact.values)

    titles = np.append(np.array([key for key in phase_dict.keys()]),['inactive'])

    datafile_RMM_split = xr.Dataset({'number':(('phase','year'), single_phase)},
                                   {'phase':titles,
                                    'year': number_per_year.time.dt.year.values
                                   })
    
    
#     datafile_RMM_split = xr.concat(single_phase, pd.Index(titles, name = 'phase'))
    
    return datafile_RMM_split





























############################ Rainfall Trends
############################ Rainfall Trends
############################ Rainfall Trends
############################ Rainfall Trends
############################ Rainfall Trends
############################ Rainfall Trends
############################ Rainfall Trends

import mystats

# Calculates the trend for each individaul grid cell
def grid_trend(x,t):
    # If every point is just a nan values. We don't want to do the polyfit calculation. Just return nan
    if np.all(np.isnan(x)):
        return float('nan')
    
    # Getting the gradient of a linear interpolation
    idx = np.isfinite(x) & np.isfinite(t) #checking where the nans are for both
    grad = np.polyfit(t[idx],x[idx],1)[0]
    return grad

def calculate_trend(percentile):
    
    # The axis number that year is
    axis_num = percentile.get_axis_num('year')
    
    '''Applying trends along each grid cell'''
    percentile_trend_meta = np.apply_along_axis(grid_trend,axis_num, percentile.values, 
                                                t = percentile.year.values)

    '''Turning into an xarray dataset'''
    # Added in logic so that now data with or without phase vlaues 
    # can be passed in. This works by creating dict with lat and lon.
    
    # List of the coordinates from the array itself
    coord_list = ['lat','lon']
    
    # The values to be used for each coordinates.
    coord_dict = {'lat':percentile.lat,'lon':percentile.lon}
      
    # If phase is also in the coord_list then we have to add this to the coord dict.
    # The reorder so that pahse is the first element in the dict.
    print(list(percentile))
    if 'phase' in list(percentile.coords):
        coord_dict['phase'] = percentile.phase.values
        coord_dict = {k:coord_dict[k] for k in ['phase','lat','lon']}
        # Adding phase too first element of coord list. 
        coord_list = ['phase'] + coord_list
    
    print('\n')
    print(coord_list)#, percentile_trend_meta.values.shape, coord_dict, sep = '\n')
    
    trend  = xr.Dataset({'trend':(coord_list, percentile_trend_meta)},
                        coord_dict)
    return trend

def convert_to_percent_per_decade(percentile, trend):
    
    mean_gridcell = percentile.mean(dim = 'year')
    
    return (trend * 10 / mean_gridcell) * 100

def calculate_pvals(percentile, trend):
    year_num = percentile.get_axis_num('year')
    
    trend_pval_meta = np.apply_along_axis(mystats.mann_kendall, year_num, percentile)

    '''Turning into an xarray dataset'''
    # Added in logic so that now data with or without phase vlaues 
    # can be passed in. This works by creating dict with lat and lon.
    
    # List of the coordinates from the array itself
    coord_list = ['lat','lon']
    
    # The values to be used for each coordinates.
    coord_dict = {'lat':percentile.lat,'lon':percentile.lon}
      
    # If phase is also in the coord_list then we have to add this to the coord dict.
    # The reorder so that pahse is the first element in the dict.
    if 'phase' in list(percentile.coords):
        coord_dict['phase'] = percentile.phase.values
        coord_dict = {k:coord_dict[k] for k in ['phase','lat','lon']}
        # Adding phase too first element of coord list.     
        coord_list = ['phase'] + coord_list

    pvals  = xr.Dataset({'pvals':(coord_list, trend_pval_meta)},
                        coord_dict) 
    
    return pvals

def significant_trend_calc(data, pvals):
    sig = data.where(np.logical_and(pvals.pvals >= 0 ,pvals.pvals <= 0.15))
    

    return sig

def return_alltrendinfo_custom(data, normalise = 0):
    import load_dataset as load

    if normalise == 'phase':
        rmm = load.load_rmm()
        rmm = wet_season_year(rmm)

        phase_count = count_in_rmm_phase(rmm)
        data = (data/phase_count.number)
        
    elif normalise == 'subphase':
        rmm = load.load_rmm()
        rmm = wet_season_year(rmm)
        subphase_count = count_in_rmm_subphase(rmm)

        data = (data/subphase_count.number)

    print('calculating trend', end = '')
    # Calculates the trend
    trend = calculate_trend(data)
    print(': complete')
    

    # Convertes to percent per decade
    print('converting to percent per decade', end = '')
    trend_percent = convert_to_percent_per_decade(data, trend)
    print(': complete')

    # Calculates the significant values
    print('finding significant points', end = '')
    pvals =  calculate_pvals(data, trend)
    print(': complete')

    print('getting just significant trend points', end = '')
    trend_sig = significant_trend_calc(trend, pvals)
    trend_percent_sig = significant_trend_calc(trend_percent, pvals)
    print(': complete')

    return trend, trend_sig, trend_percent, trend_percent_sig