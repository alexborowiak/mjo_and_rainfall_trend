import xarray as xr
import numpy as np
import pandas as pd
import sys
from scipy.stats import spearmanr

'''Counts the number of days in each of the MJO phases for each wet-season. This is useful for 
normalising all of the count trends'''
def count_in_rmm_phase(rmm):
    


    
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

    phase_count = xr.Dataset({'number':(('phase','year'), single_phase)},
                                   {'phase':titles,
                                    'year': number_per_year.time.dt.year.values
                                   })
    
    

    
    return phase_count


def resample_phase_to_subphase(data):
    
    enhanced = data.sel(phase = ['4','5','6']).sum(dim = 'phase')
    suppressed = data.sel(phase = ['1','2','3']).sum(dim = 'phase')
    trans = data.sel(phase = ['4','8']).sum(dim = 'phase')
    inact = data.sel(phase = 'inactive').drop('phase')
    
    return xr.concat([enhanced,suppressed, trans, inact], 
                     pd.Index(['enhanced','suppressed','transition','inacitve'], name = 'phase'))




'''~~~~~~~~~~~~~~~~~~~~~~~~~~ CORRELATION'''
# This function is from Scott, and is for appyling np.apply_along_axis
# for two arrays so that spearman rank correlation fucntion can be performed

def helper(x, len_a):
    # split x into 'a' and 'b' parts
    
    xa = x[0:len_a]
    xb = x[len_a:]
    
    corr,sig = spearmanr(xa,xb)
    
    return corr, sig

def spearman_correlation_rmm(awap, rmm):
    
    # THis is concating the two different datasets into the one xarry file. They will be split down the middle
    # later on
    len_a = awap.dims['year']
    index_concat = xr.concat([awap,rmm], dim = 'year')

    axis =  index_concat.number.get_axis_num('year')
    arr = index_concat.number
    spearman_index_meta, sig_meta = np.apply_along_axis(helper, arr = arr, 
                                         axis = axis, len_a = len_a)

    spearman_index = xr.Dataset({'precip':(('lat','lon'), spearman_index_meta)},{
        'lat':awap.lat.values,
        'lon':awap.lon.values
    })
    
    
    spearman_sig = xr.Dataset({'precip':(('lat','lon'), sig_meta)},{

        'lat':awap.lat.values,
        'lon':awap.lon.values
    })
    
    # Getting only the significant points
    spearman_sig = spearman_sig.where(spearman_sig.precip < 0.1)
    
    return spearman_index, spearman_sig