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



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~ANOMALY CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following functions all work together to calculate the anomalies in a given phase of the MJO'''



# This function readis in the RMM form the Bureau's website and turns it into an 
# xarrau file. The useragent will need to be changed depending on the computer. Currently it is set to the VDI

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''




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






'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''



def  calculate_anomalies_1to8_mean(variable_split, variable):
    phase_mean = variable_split.groupby('phase').mean(dim = 'time')
    overall_mean = variable.mean(dim = 'time')
    anomalies = phase_mean/overall_mean
    
    return phase_mean, anomalies



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


def  calculate_anomalies_1to8_percentile(variable_split, variable, q):
    phase_mean = variable_split.groupby('phase').reduce(np.nanpercentile, q = q, dim = 'time')
    overall_mean = variable.reduce(np.nanpercentile, q = q, dim = 'time')
    anomalies = phase_mean/overall_mean
    
    return phase_mean, anomalies

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''


def calculate_1to8_anomalies_for_variables(variable, anomaly_type = 'mean'):
    
    #Read in RMM
    rmm = load_rmm()
    
    # Split Via RMM
    variable_split = split_into_1to8(variable, rmm)
    
    # Calculate anomalies
    if anomaly_type == 'mean':
        variable_values, variable_anomalies = calculate_anomalies_1to8_mean(variable_split, variable)
    else:
        variable_values, variable_anomalies = calculate_anomalies_1to8_percentile(variable_split, variable, anomaly_type) 
    
    
    return  variable_values, variable_anomalies





'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Anomly Functions'''

def sum_and_anomly(data_split, data, rmm, calc_anomaly = 1):
    # * calc_anomaly = 0 is for ACCESS-S when you want to calculate the anomaly at
    # the end
    
    
    # This function returns the percent of rainfall events falling in each phases.
    # The total number of events is not ideal, as there are different number of
    # events in each phase, thus favouring the inacive phase where the MJO is ~ 50%
    # of the time.
    
    #Count the number of events in each MJO phase
    data_sum = data_split.groupby('phase').sum(dim = 'time')
    
    # Counting the number of days in each phase each year, then summing this up.
    rmm_act = rmm.where(rmm.amplitude >=1 , drop = True)
    rmm_act_count = rmm_act.groupby('phase').count(dim = 'time')
    rmm_act_count['phase'] = rmm_act_count.phase.values.astype(int).astype(str)
    rmm_act_count = rmm_act_count.rename({'amplitude':'number'})


    rmm_inact_count = rmm.where(rmm.amplitude < 1, drop = True).count(dim = 'time')\
    .rename({'phase':'number'}).drop('amplitude')
    rmm_inact_count['phase'] = ['inactive']

    rmm_count = xr.concat([rmm_act_count, rmm_inact_count ], dim = 'phase')
    
    # Normalising by the number of days in each phase, then converting to a percenti
    data_sum_norm = (data_sum.precip * 100/rmm_count.number).to_dataset(name = 'precip')
    
    
    ######### Anomalies. This is all put into an if statement as for ACCESS-S this be done at
    # the end
    if calc_anomaly:
        
        # The total number of rainfall events
        total_sum = data.sum(dim = 'time')
        
        # The total number of rainfall events/ the total number of days (the average percent of rain days)
        total_sum_norm = total_sum * 100/ rmm.amplitude.count(dim = 'time') 
        
        
        # Comparing to the climatology
        sum_anomaly = data_sum_norm/total_sum_norm
        

        return data_sum_norm, sum_anomaly
    
    
    return data_count_norm



def count_and_anomly(data_split, data, rmm, calc_anomaly = 1):
    # * calc_anomaly = 0 is for ACCESS-S when you want to calculate the anomaly at
    # the end
    
    
    # This function returns the percent of rainfall events falling in each phases.
    # The total number of events is not ideal, as there are different number of
    # events in each phase, thus favouring the inacive phase where the MJO is ~ 50%
    # of the time.
    
    #Count the number of events in each MJO phase
    data_count = data_split.groupby('phase').count(dim = 'time')
    
    # Counting the number of days in each phase each year, then summing this up.
    rmm_act = rmm.where(rmm.amplitude >=1 , drop = True)
    rmm_act_count = rmm_act.groupby('phase').count(dim = 'time')
    rmm_act_count['phase'] = rmm_act_count.phase.values.astype(int).astype(str)
    rmm_act_count = rmm_act_count.rename({'amplitude':'number'})


    rmm_inact_count = rmm.where(rmm.amplitude < 1, drop = True).count(dim = 'time')\
    .rename({'phase':'number'}).drop('amplitude')
    rmm_inact_count['phase'] = ['inactive']

    rmm_count = xr.concat([rmm_act_count, rmm_inact_count ], dim = 'phase')
    
    # Normalising by the number of days in each phase, then converting to a percenti
    data_count_norm = (data_count.precip * 100/rmm_count.number).to_dataset(name = 'precip')
    
    
    ######### Anomalies. This is all put into an if statement as for ACCESS-S this be done at
    # the end
    if calc_anomaly:
        
        # The total number of rainfall events
        total_count = data.count(dim = 'time')
        
        # The total number of rainfall events/ the total number of days (the average percent of rain days)
        total_count_norm = total_count * 100/ rmm.amplitude.count(dim = 'time') 
        
        
        # Comparing to the climatology
        count_anomaly = data_count_norm/total_count_norm
        

        return data_count_norm, count_anomaly
    
    
    return data_count_norm









def count_and_anomly_month(data_split_total, data_total, rmm_total):
    # * calc_anomaly = 0 is for ACCESS-S when you want to calculate the anomaly at
    # the end
    
    import datetime as dt
    # This function returns the percent of rainfall events falling in each phases.
    # The total number of events is not ideal, as there are different number of
    # events in each phase, thus favouring the inacive phase where the MJO is ~ 50%
    # of the time.
    count_stor = []
    anom_stor = []
    months = [10,11,12,1,2,3]
    for month in months:
        
        data_split = data_split_total.where(data_split_total.time.dt.month == month, drop = True)
        data = data_total.where(data_total.time.dt.month == month, drop = True)
        rmm = rmm_total.where(rmm_total.time.dt.month == month, drop = True)
        
    
        #Count the number of events in each MJO phase
   
        data_count = data_split.groupby('phase').count(dim = 'time')

        # Counting the number of days in each phase each year, then summing this up.
        rmm_act = rmm.where(rmm.amplitude >=1 , drop = True)
        rmm_act_count = rmm_act.groupby('phase').count(dim = 'time')
        rmm_act_count['phase'] = rmm_act_count.phase.values.astype(int).astype(str)
        rmm_act_count = rmm_act_count.rename({'amplitude':'number'})


        rmm_inact_count = rmm.where(rmm.amplitude < 1, drop = True).count(dim = 'time')\
        .rename({'phase':'number'}).drop('amplitude')
        rmm_inact_count['phase'] = ['inactive']

        rmm_count = xr.concat([rmm_act_count, rmm_inact_count ], dim = 'phase')

        # Normalising by the number of days in each phase, then converting to a percent
        '''This is the first thing to be stored'''
        data_count_norm = (data_count.precip * 100/rmm_count.number).to_dataset(name = 'precip')


        ######### Anomalies. This is all put into an if statement as for ACCESS-S this be done at
        # the end

        # The total number of rainfall events
        total_count = data.count(dim = 'time')

        # The total number of rainfall events/ the total number of days (the average percent of rain days)
        total_count_norm = total_count * 100/ rmm.amplitude.count(dim = 'time') 


        # Comparing to the climatology
        '''This is the second thing to be stored'''
        count_anomaly = data_count_norm/total_count_norm
        
        
        count_stor.append(data_count_norm)
        anom_stor.append(count_anomaly)
        
        
    data_count_norm_tot = xr.concat(count_stor, pd.Index(months, name = 'month'))   
    count_anomaly_tot = xr.concat(anom_stor, pd.Index(months, name = 'month'))  
    
    return data_count_norm_tot, count_anomaly_tot
    
    










def count_and_anomly_season(data_split_total, data_total, rmm_total):
    # * calc_anomaly = 0 is for ACCESS-S when you want to calculate the anomaly at
    # the end
    
    import datetime as dt
    # This function returns the percent of rainfall events falling in each phases.
    # The total number of events is not ideal, as there are different number of
    # events in each phase, thus favouring the inacive phase where the MJO is ~ 50%
    # of the time.
    count_stor = []
    anom_stor = []
    months = [[10,11,12],[1,2,3]]
    for month in months:
        
        data_split = data_split_total.where(data_split_total.time.dt.month == month, drop = True)
        data = data_total.where(data_total.time.dt.month.isin(month), drop = True)
        rmm = rmm_total.where(rmm_total.time.dt.month.isin(month), drop = True)
        
    
        #Count the number of events in each MJO phase
   
        data_count = data_split.groupby('phase').count(dim = 'time')

        # Counting the number of days in each phase each year, then summing this up.
        rmm_act = rmm.where(rmm.amplitude >=1 , drop = True)
        rmm_act_count = rmm_act.groupby('phase').count(dim = 'time')
        rmm_act_count['phase'] = rmm_act_count.phase.values.astype(int).astype(str)
        rmm_act_count = rmm_act_count.rename({'amplitude':'number'})


        rmm_inact_count = rmm.where(rmm.amplitude < 1, drop = True).count(dim = 'time')\
        .rename({'phase':'number'}).drop('amplitude')
        rmm_inact_count['phase'] = ['inactive']

        rmm_count = xr.concat([rmm_act_count, rmm_inact_count ], dim = 'phase')

        # Normalising by the number of days in each phase, then converting to a percent
        '''This is the first thing to be stored'''
        data_count_norm = (data_count.precip * 100/rmm_count.number).to_dataset(name = 'precip')


        ######### Anomalies. This is all put into an if statement as for ACCESS-S this be done at
        # the end

        # The total number of rainfall events
        total_count = data.count(dim = 'time')

        # The total number of rainfall events/ the total number of days (the average percent of rain days)
        total_count_norm = total_count * 100/ rmm.amplitude.count(dim = 'time') 


        # Comparing to the climatology
        '''This is the second thing to be stored'''
        count_anomaly = data_count_norm/total_count_norm
        
        
        count_stor.append(data_count_norm)
        anom_stor.append(count_anomaly)
        
    splits = ['OND','JFM']
    data_count_norm_tot = xr.concat(count_stor, pd.Index(splits, name = 'month'))   
    count_anomaly_tot = xr.concat(anom_stor, pd.Index(splits, name = 'month'))  
    
    return data_count_norm_tot, count_anomaly_tot
    
        
    
    
    
    
    
    
    
    
    
    


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~TREND CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''



'''Counts the number of days in each of the MJO phases for each wet-season. This is useful for 
normalising all of the count trends'''
def count_in_rmm_phase(rmm):
    
    # We are looking at wet season, so will need to be made into wet season data set
    rmm = wet_season_year(rmm)
    print(len(np.unique(rmm.time.dt.year.values)))
    print('---')
    rmm_act = rmm.where(rmm.amplitude > 1, drop = True)
    
    phases = np.arange(1,9)
    single_phase = []
    for phase in phases:

         # Just the data for this single rmm phase
        rmm_single_phase = rmm_act.where(rmm_act.phase == phase, drop = True)
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
    
    
#     datafile_RMM_split = xr.concat(single_phase, pd.Index(titles, name = 'phase'))
    
    return datafile_RMM_split


def count_in_rmm_subphase(rmm):
    
    # We are looking at wet season, so will need to be made into wet season data set
    rmm = wet_season_year(rmm)
    
    enhanced = [5,6,7]
    suppressed = [1,2,3]
    transition = [4,8]

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




'''Each year is a wet season'''

# This function moves the start of the wet season [10, 11, 12] to the next year. This means that
# this year is just the data for one wet season

def wet_season_year(data):
    
    # This is the start of the wet_season, wet want to move it to the next year so that the start of the
    # wet season and the end are both in the one year. This makes it easier for calculatins later on 
    
    data_start = data.where(data.time.dt.month.isin([10,11,12]), drop = True) # The later months of the year
    data_start['time'] = data_start.time + pd.to_timedelta('365day') # moving them forward a year
    
    data_end = data.where(data.time.dt.month.isin([1,2,3]), drop = True) # The end half
    
    total = data_end.combine_first(data_start) # All in one year now :)
    
    return total






sys.path.append('/home/563/ab2313/MJO/functions')
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
    percenilte_trend_meta = np.apply_along_axis(grid_trend,axis_num, percentile.values, 
                                                t = percentile.year.values)

    '''Turning into an xarray dataset'''
    trend  = xr.Dataset(
        {'trend':(('phase','lat','lon'), percenilte_trend_meta)},

        {
        'phase':percentile.phase.values, 
         'lat':percentile.lat,
        'lon':percentile.lon}
    )
    
    
    
    return trend






def convert_to_percent_per_decade(percentile, trend):
    
    mean_gridcell = percentile.mean(dim = 'year')
    
    
    return (trend * 10 / mean_gridcell) * 100





def calculate_pvals(percentile, trend):
    year_num = percentile.get_axis_num('year')
    
    trend_pval_meta = np.apply_along_axis(mystats.mann_kendall, year_num, percentile)


    pvals  = xr.Dataset(
        {'pvals':(('phase','lat','lon'), trend_pval_meta)},

        {
        'phase':percentile.phase.values, 
         'lat':percentile.lat,
        'lon':percentile.lon}
    )
    
    
    return pvals




def significant_trend_cacl(data, pvals):
    sig = data.where(np.logical_and(pvals.pvals >= 0 ,pvals.pvals <= 0.05  ))
    

    return sig



def return_alltrendinfo_custom(data, normalise = 0):
    import load_dataset as load

    if normalise == 'phase':
        rmm = load.load_rmm()

        subphase_count = count_in_rmm_phase(rmm) # This included a wet-season recalibration
        data = (data/subphase_count.number)
        
    
    elif normalise == 'subphase':
        rmm = load.load_rmm()
        subphase_count = count_in_rmm_subphase(rmm) # This included a wet-season recalibration
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
    trend_sig = significant_trend_cacl(trend, pvals)
    trend_percent_sig = significant_trend_cacl(trend_percent, pvals)
    print(': complete')

    return trend, trend_sig, trend_percent, trend_percent_sig



def return_alltrendinfo(data,q = 90):
    #Calculated the percentile
    # The percentiles of each year. Maintains MJO splits
    if type(q) == int:
        percentile = data.groupby('time.year').reduce(np.nanpercentile, dim = 'time', q = q)
    elif q == 'mean':
        percentile = data.groupby('time.year').mean(dim = 'time')
    elif q == 'all':
        pass
    
    percentile = percentile.to_array().squeeze()
    print('percentile/mean has been calculated')

#     Calculates the trend
    trend = calculate_trend(percentile)
    
    print('trend has been calculated')
    
    
    # Convertes to percent per decade
    trend_percent = convert_to_percent_per_decade(percentile, trend)
    print('trend has been converted to percent')
    
    # Calculates the significant values
    pvals =  calculate_pvals(percentile, trend)
    trend_sig = significant_trend_cacl(trend, pvals)
    trend_percent_sig = significant_trend_cacl(trend_percent, pvals)
    print('significant points habe been found')
    print('function is complete')

    return percentile, trend, trend_sig, trend_percent, trend_percent_sig
    
    
