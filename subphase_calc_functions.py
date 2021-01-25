import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.array
import cartopy.crs as ccrs
import pickle
import matplotlib.colors as colors
import datetime as dt
import pickle
from matplotlib.colors import BoundaryNorm
import glob
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib.gridspec as gridspec




'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~RAW CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''





def split_into_subphase(datafile, rmm):
    
    # This function splits the MJO data into the enhanced, suppressed and tranistion MJO phases
    # as defined by Murppy et al. 2016.
    
    enhanced = [4,5,6]
    suppressed = [1,2,8]
    transition = [3,7]

    phase_dict =  {'enhanced': enhanced, 'suppressed': suppressed, 'transition': transition}
    single_phase = []
    rmm_act = rmm.where(rmm.amplitude > 1, drop = True)


    for phase_name, phase_nums in phase_dict.items():

         # The dates of this phase
        rmm_single_dates = rmm_act.where(rmm_act.phase.isin(phase_nums), drop = True).time.values
         # The datafile data in this phase
        datafile_single = datafile.where(datafile.time.isin(rmm_single_dates), drop = True)
        # Appending
        single_phase.append(datafile_single) 


    '''Inactive Phase'''
    rmm_inact_dates = rmm.where(rmm.amplitude <=1 , drop = True).time.values
    inactive_datafile = datafile.where(datafile.time.isin(rmm_inact_dates), drop = True)

    single_phase.append(inactive_datafile)     

    titles = np.append(np.array([key for key in phase_dict.keys()]),['inactive'])

    datafile_RMM_split = xr.concat(single_phase, pd.Index(titles, name = 'phase'))
    
    return datafile_RMM_split



def split_into_enso(data, nino34):
    el_nino_dates = nino34.where(nino34.nino34 > 0, drop = True).time.values
    la_nina_dates = nino34.where(nino34.nino34 <= 0, drop = True).time.values

    
    # Now just getting the data for the El Nino and La Nina Dates
    data_elnino = data.where(data.time.isin(el_nino_dates))
    data_lanina = data.where(data.time.isin(la_nina_dates))

#     return data_elnino, data_lanina
    # Putting them both into the one xarray
    data_nino = xr.concat([data_elnino, data_lanina], pd.Index(['el nino', 'la nina'], name = 'nino'))
    
    return data_nino




'''Returns the events that are above a specific percentile'''

def find_events_above_q(raw, split, q):
    
    # These are the q-values for each month in each location. The threshold
    pval = raw.groupby('time.month').reduce(np.nanpercentile, q = q, dim = 'time')
    
    
    storage = []
   
    for month in [10,11,12,1,2,3]:

        # qth percentile for each grid cell of that month
        month_pval = pval.sel(month = month)

        # These are the events that are just in the month in question
        data_month = split.where(split.time.dt.month == month, drop = True)

        # These are the events above the percentile that we are looking at
        data_pval = data_month.where(data_month >= month_pval, drop = True)
        

        storage.append(data_pval)
    
    # Merging everything back together
    above_q = xr.concat(storage, dim = 'time').to_dataset(name  = 'precip').squeeze()
    
    return above_q

'''This function was created as the other 'find_events_above_q' didn't work when
splitting via both MJO and ENSO. This is also more simple to understand, getting the events 
above a certain percentile then splitting after that'''
def unsplit_find_events_above_q(data, q):
    
    # Findint the qth percentile of each month for the dataset
    pval = data.groupby('time.month').reduce(np.nanpercentile, q = q, dim = 'time')
    
    
    storage = []
    # Looping through each month and getting the events above the qth percentile
    for month in [10,11,12,1,2,3]:
        
        # The raw data that is just in the month in question
        data_month = data.where(data.time.dt.month == month, drop = True)
        
        # The qth percentile for the month in question
        pval_month = pval.sel(month =  month)
        
        # The data for this month that is above the percentile
        data_pval = data_month.where(data_month.precip > pval_month.precip, drop = True)
        
        # Storing to be put back into xarray form below
        storage.append(data_pval)
    
    # Back into one continuos xararay file
    above_q = xr.concat(storage, dim = 'time').drop('month')
    
    return above_q












'''Counts the number of days in each of the MJO subphase for each wet-season. This is useful for 
normalising all of the count trends'''
def count_in_rmm_subphase_monthly(rmm_total):
    
    month_stor = []
    months = [10,11,12,1,2,3]
    for month in months:
        
        rmm = rmm_total.where(rmm_total.time.dt.month == month, drop = True)
        
        
        # New
#         enhanced = [4,5,6,7]
#         suppressed  =[1,2]
#         transition = [3,8]
        
        
#           Old 
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
            number_per_phase = rmm_single_phase.phase.count(dim = 'time')
            # Appending
            single_phase.append(number_per_phase.values)



        '''Inactive Phase'''
        rmm_inact = rmm.where(rmm.amplitude <=1)# , drop = True)
        number_per_inact = rmm_inact.phase.count(dim = 'time')

        single_phase.append(number_per_inact.values)

        titles = np.append(np.array([key for key in phase_dict.keys()]),['inactive'])

        datafile_RMM_split = xr.Dataset({'number':(('phase'), single_phase)},
                                       {'phase':titles
                                       })
        
        month_stor.append(datafile_RMM_split)
            
    total_rmm_count = xr.concat(month_stor, pd.Index(months, name = 'month'))

#     datafile_RMM_split = xr.concat(single_phase, pd.Index(titles, name = 'phase'))
    
    return total_rmm_count



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~ANOMALY CALCULATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The following functions all work together to calculate the anomalies in a given phase of the MJO'''

def count_anomalies(raw, split, q):
    
    
    # Thr events above the percentile
    above_q = find_events_above_q(raw, split, q)
    
    # The number of events above the percentile for each phase
    above_q_count = above_q.groupby('time.month').count(dim = 'time')
    
    # The total number of events for each phase
    split_count = split.groupby('time.month').count(dim = 'time')
    
    
    # The anomaly calculation
    per = (100 - q)/100
    anomaly = (above_q_count/ split_count)/per
    
    
    return anomaly



def enso_count_anomalies(raw, q,nino34, rmm):
    
    '''Above q events'''
    # The events above the percentile
    above_q = unsplit_find_events_above_q(raw, q)
    # MJO splitting
    above_q_split  = split_into_subphase(above_q, rmm)
    # ENSO splitting
    above_q_split_2  = split_into_enso(above_q_split, nino34)
    # The number of events above the percentile for each phase
    above_q_count = above_q_split_2.groupby('time.month').count(dim = 'time')
    
    '''All events'''
    # MJO splitting 
    raw_split  = split_into_subphase(raw, rmm)
    # ENSO splitting
    raw_split_2  = split_into_enso(raw_split, nino34)
    # The total number of events
    raw_count = raw_split_2.groupby('time.month').count(dim = 'time')
    
    '''Calculation'''
    # The anomaly calculation
    per = (100 - q)/100
    anomaly = (above_q_count/ raw_count)/ per
    
    return anomaly



def int_anomalies(raw, split, q):
    
    
    # The events above the percentile
    above_q = find_events_above_q(raw, split, q)
    
    # The mean of events above the percentile for each phase
    above_q_int_split = above_q.groupby('time.month').mean(dim = 'time')
    
    # The mean of all events for each phase
    above_q_int = above_q.groupby('time.month').mean(dim = ['time','phase'])
    
    # The anomaly calculation
    anomaly = above_q_int_split/above_q_int
    
    
    return anomaly

def enso_int_anomalies(raw, q,nino34, rmm):
    
    '''Above q events'''
    # The events above the percentile
    above_q = unsplit_find_events_above_q(raw, q)
    # MJO splitting
    above_q_split  = split_into_subphase(above_q, rmm)
    # ENSO splitting
    above_q_split_2  = split_into_enso(above_q_split, nino34)
    # The mean intensity of events above the percentile for each phase
    above_q_split_2_mean = above_q_split_2.groupby('time.month').mean(dim = 'time')
    
    '''All events above q'''
    above_q_mean = above_q.groupby('time.month').mean(dim = 'time')
    
    
    '''Calculation'''
    # The anomaly calculation
    anomaly = above_q_split_2_mean/above_q_mean
    
    return anomaly

def sum_anomalies(raw, split, q):
    
    
    # The events above the percentile
    above_q = find_events_above_q(raw, split, q)
    
    
    '''Phase Split'''
    # The sum of events above the percentile for each phase
    above_q_sum_split = above_q.groupby('time.month').sum(dim = 'time')
    
    # The sum of all events for each percentile
    sum_split = split.groupby('time.month').sum(dim = 'time')
    
    # The proportion of rainfall as extreme in each phase
    sum_phase_prop = above_q_sum_split/sum_split
    
    
    ''''All Events'''
    
    # The mean of all events for each phase
    above_q_sum = above_q.groupby('time.month').sum(dim = ['time','phase'])
    
   # The sum of all events for each percentile
    sum_ = split.groupby('time.month').sum(dim = ['time','phase'])
    
    sum_prop = above_q_sum/sum_
    
    
    # The anomaly calculation
    anomaly = sum_phase_prop/sum_prop
    
    
    return anomaly




def enso_sum_anomalies(raw, q,nino34, rmm):
    
    '''Above q events'''
    # The events above the percentile
    above_q = unsplit_find_events_above_q(raw, q)
    # MJO splitting
    above_q_split  = split_into_subphase(above_q, rmm)
    # ENSO splitting
    above_q_split_2  = split_into_enso(above_q_split, nino34)
    # The sum of events above the percentile for each phase
    above_q_split_2_sum = above_q_split_2.groupby('time.month').sum(dim = 'time')
    
    '''All events above q'''
    all_split  = split_into_subphase(raw, rmm)
    # ENSO splitting
    all_split_2  = split_into_enso( all_split, nino34)
    # The sum of events above the percentile for each phase
    all_split_2_sum = all_split_2.groupby('time.month').sum(dim = 'time')
    
    '''Calculation'''
    # The proportion of events falling as extreme in each phases
    sum_phase_prop = above_q_split_2_sum/all_split_2_sum
    
    # The proportion of events falling as extreme overall
    sum_prop = above_q.groupby('time.month').sum(dim = 'time')/raw.groupby('time.month').sum(dim = 'time')

    # The anomaly calculation
    anomaly = sum_phase_prop/sum_prop
    
    return anomaly




'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~Bootstrapping Functions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''



def return_year_count(data):
    
    # This function loops through all of the months and gets the number of events each year.
    
    stor = []
    
    for month in [10,11,12,1,2,3]:
        
        # Just looking at one month
        data_month = data.where(data.time.dt.month == month, drop = True)
        
        # Counting the number of items each year
        count_month = data_month.resample(time = 'y').count(dim = 'time')
        
        # Making the index just the year values not year, month, day
        count_month['time'] = count_month.time.dt.year.values
        
        # Renaming the index
        count_month = count_month.rename({'time':'year'})
        
        # Storing
        stor.append(count_month)
    
    # Returning as an xarray concat with year and month as seperate indices
    return xr.concat(stor, pd.Index([10,11,12,1,2,3], name = 'month')) 




def return_year_sum(data):
    
    # This function loops through all of the months and gets the number of events each year.
    
    stor = []
    
    for month in [10,11,12,1,2,3]:
        
        # Just looking at one month
        data_month = data.where(data.time.dt.month == month, drop = True)
        
        # Counting the number of items each year
        sum_month = data_month.resample(time = 'y').sum(dim = 'time')
        
        # Making the index just the year values not year, month, day
        sum_month['time'] = sum_month.time.dt.year.values
        
        # Renaming the index
        sum_month = sum_month.rename({'time':'year'})
        
        # Storing
        stor.append(sum_month)
    
    # Returning as an xarray concat with year and month as seperate indices
    return xr.concat(stor, pd.Index([10,11,12,1,2,3], name = 'month')) 




# This function return jsut the significant points.

def sel_sig_all(normal, boot):


    # Both 5th and 95th percentile >= 1 (from bootdata)
    station_sig_pos = normal.precip.where(
        np.logical_and(
            boot.isel(percentile = 0).precip >= 1, 
            boot.isel(percentile = 1).precip >= 1)
            )
    

    # Both 5th and 95th percentile <= 1
    station_sig_neg = normal.precip.where(
        np.logical_and(
            boot.isel(percentile = 0).precip <= 1, 
            boot.isel(percentile = 1).precip <= 1)
        )
    
    # Using scotts code to combine this into one array:
        # If the pos is finite then use the value from pos, else use the value form negative
    total = xr.where(np.isfinite(station_sig_pos),station_sig_pos, station_sig_neg)
    
    # Reutrns as a dataset otherwise, want a datarray
    total = total.to_dataset(name = 'precip')
    
    return total




def year_resmaples(data):
    
    count_stor = []
    sum_stor = []
    
    for month in [10,11,12,1,2,3]:
    
        data_month = data.where(data.time.dt.month == month,drop = True)

        # The number of events above the percentile for each phase
        data_month_count = data_month.resample(time = 'y').count(dim = 'time')
        data_month_sum = data_month.resample(time = 'y').sum(dim = 'time')


         # format is now just year rather than day,month, year
        data_month_count['time'] = data_month_count.time.dt.year.values

        data_month_sum['time'] = data_month_sum.time.dt.year.values
        
        # Changing dim name to conform to previous conventions
        data_month_count = data_month_count.rename({'time':'year'})
        
        data_month_sum = data_month_sum.rename({'time':'year'})
        
        
        # Storing all the restuls
        count_stor.append(data_month_count)
   
        sum_stor.append(data_month_sum)



    count_total = xr.concat(count_stor, pd.Index([10,11,12,1,2,3], name = 'month'))

    sum_total = xr.concat(sum_stor, pd.Index([10,11,12,1,2,3], name = 'month'))
    
    return count_total, sum_total







def return_year_count_month_split(raw, q, rmm):
    
    '''Above q events'''
    # The events above the percentile
    above_q = unsplit_find_events_above_q(raw, q)
    # MJO splitting
    above_q_split  = split_into_subphase(above_q, rmm)
    
    
    '''All events'''
    # MJO splitting 
    raw_split  = split_into_subphase(raw, rmm)
    
    '''Resampling'''
    
    q_stor = []
    raw_stor = []
    
    for month in [10,11,12,1,2,3]:
    
        above_q_split_month = above_q_split.where(above_q_split.time.dt.month == month,drop = True)
        raw_split_month = raw_split.where(raw_split.time.dt.month == month, drop = True)
        
        # The number of events above the percentile for each phase
        above_q_count = above_q_split_month.resample(time = 'y').count(dim = 'time')

        raw_count = raw_split_month.resample(time = 'y').count(dim = 'time')
        

         # format is now just year rather than day,month, year
        above_q_count['time'] = above_q_count.time.dt.year.values

        raw_count['time'] = raw_count.time.dt.year.values
        
        # Changing dim name to conform to previous conventions
        above_q_count = above_q_count.rename({'time':'year'})
        
        
        raw_count = raw_count.rename({'time':'year'})
        
        
        q_stor.append(above_q_count)
   
        raw_stor.append(raw_count)



    q_total = xr.concat(q_stor, pd.Index([10,11,12,1,2,3], name = 'month'))

    raw_total = xr.concat(raw_stor, pd.Index([10,11,12,1,2,3], name = 'month'))
    
    return q_total, raw_total


def return_year_count(raw, q, rmm):
    
    '''Above q events'''
    # The events above the percentile
    above_q = unsplit_find_events_above_q(raw, q)
    # MJO splitting
    above_q_split  = split_into_subphase(above_q, rmm)

    # The number of events above the percentile for each phase
    above_q_count = above_q_split.resample(time = 'y').count(dim = 'time')
    
     # format is now just year rather than day,month, year
    above_q_count['time'] = above_q_count.time.dt.year.values
    
    # Changing dim name to conform to previous conventions
    above_q_count = above_q_count.rename({'time':'year'})
    

   
    
    '''All events'''
    # MJO splitting 
    raw_split  = split_into_subphase(raw, rmm)

    # The total number of events
    raw_count = raw_split.resample(time = 'y').count(dim = 'time')
    
    # format is now just year rather than day,month, year
    raw_count['time'] = raw_count.time.dt.year.values
    
    # Changing dim name to conform to previous conventions
    raw_count = raw_count.rename({'time':'year'})

    
    return above_q_count,raw_count


def return_year_count_enso(raw, q,nino34, rmm):
    
    '''Above q events'''
    # The events above the percentile
    above_q = unsplit_find_events_above_q(raw, q)
    # MJO splitting
    above_q_split  = split_into_subphase(above_q, rmm)
    # ENSO splitting
    above_q_split_2  = split_into_enso(above_q_split, nino34)
    # The number of events above the percentile for each phase
    above_q_count = above_q_split_2.resample(time = 'y').count(dim = 'time')
    
     # format is now just year rather than day,month, year
    above_q_count['time'] = above_q_count.time.dt.year.values
    
    # Changing dim name to conform to previous conventions
    above_q_count = above_q_count.rename({'time':'year'})
    

    
    
    '''All events'''
    # MJO splitting 
    raw_split  = split_into_subphase(raw, rmm)
    # ENSO splitting
    raw_split_2  = split_into_enso(raw_split, nino34)
    # The total number of events
    raw_count = raw_split_2.resample(time = 'y').count(dim = 'time')
    
    # format is now just year rather than day,month, year
    raw_count['time'] = raw_count.time.dt.year.values
    
    # Changing dim name to conform to previous conventions
    raw_count = raw_count.rename({'time':'year'})


    
    return above_q_count,raw_count



def return_year_int_enso(raw, q,nino34, rmm):
    
    '''Above q events'''
    # The events above the percentile
    above_q = unsplit_find_events_above_q(raw, q)
    
    '''MJO and ENSO SPlitting'''
    # MJO splitting
    above_q_split  = split_into_subphase(above_q, rmm)
    # ENSO splitting
    above_q_split_2  = split_into_enso(above_q_split, nino34)
    
    '''Split Calcs'''
    phase_sum = above_q_split_2.resample(time = 'y').sum(dim = 'time')
    phase_sum['time'] = phase_sum.time.dt.year.values
    
    phase_count = above_q_split_2.resample(time = 'y').count(dim = 'time')
    phase_count['time'] = phase_count.time.dt.year.values
    
    '''All events Calcsl'''
    all_sum = above_q.resample(time = 'y').sum(dim = 'time')
    all_sum['time'] = all_sum.time.dt.year.values
    
    
    all_count = above_q.resample(time = 'y').count(dim = 'time')
    all_count['time'] = all_count.time.dt.year.values

    return phase_sum,phase_count,all_sum,all_count
