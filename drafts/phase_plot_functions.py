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
rb = plt.cm.RdBu
bm = plt.cm.Blues
best_blue = '#9bc2d5'
recherche_red = '#fbc4aa'
wondeful_white = '#f8f8f7'
import glob
import pdb

import warnings
warnings.filterwarnings('ignore')

import matplotlib.gridspec as gridspec

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~RAW VALUE PLOTS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def upper_low_bound(vmin, vmax):
    
    if vmin == 0:
        lower_bound = 0
    else:
        low_mag = OrderOfMagnitude(vmin) * 10
        lower_bound = np.floor(vmin/low_mag) * low_mag
        
        
    if vmax == 0:
        upper_bound = 0
    else:
        
        high_mag = OrderOfMagnitude(vmax) * 10
        upper_bound = np.ceil(vmax/high_mag) * high_mag
    
    
    return lower_bound, upper_bound
    

def OrderOfMagnitude(number):
    import math
    mag = math.floor(math.log(number, 10))   
    if mag == 0:
        return 0.1
    else:
        return mag
     
    
def values_plots(datafile, title = '', cbar_title = '',cbar_num_steps = 10,  savefig = 0 , savedir = '',
                 cmap = plt.cm.Blues, ceil = 1):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc

    vmax = datafile.reduce(np.nanpercentile, q= 95,dim = ['lon','lat','phase'])
    vmin = datafile.reduce(np.nanpercentile, q= 5,dim = ['lon','lat','phase'])
    
    lower_bound, upper_bound = upper_low_bound(vmin, vmax)
    
    if lower_bound == 1:
        lower_bound = 0

    fig = plt.figure(figsize = (24,12))
    gs = gridspec.GridSpec(4,3,hspace = 0.5, wspace = 0, height_ratios = [0.2, 1,1,1])
    fig.suptitle(title, fontsize = 35, y = 0.97)

    phases = np.append(np.arange(1,9),'inactive')
    

    bounds = np.linspace(lower_bound, upper_bound, cbar_num_steps )
    if ceil:
        bounds = np.ceil(bounds)

#     norm = BoundaryNorm(bounds, cmap.N)
    
    
    for i,phase in enumerate(phases):
 
        ax = fig.add_subplot(gs[i + 3], projection  = ccrs.PlateCarree())
    
        data = datafile.sel(phase = str(phase))
        
        pdata = data.plot(ax =ax,cmap = cmap, add_colorbar = False, vmin = lower_bound, vmax = upper_bound)

        #Removing the spines of the plot. Cartopy requires different method
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        if phase == 'inactive':
            ax.set_title('Inactive Phase', size = 25)
        else:
            ax.set_title('Phase ' + str(phase), size = 25)
        
        '''~~~~~ Colorbar'''

    axes = plt.subplot(gs[0,:])
    cbar = plt.colorbar(pdata, cax = axes,boundaries = bounds, orientation = 'horizontal')
    cbar.ax.set_title(cbar_title, size = 20);
    cbar.ax.set_xticklabels(bounds.astype(str), fontsize = 15)
    
    
    
    if savefig:
        fig.savefig(savedir + title + '.png', dpi = 400)
    
    
    
    
    
    
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~ANOMALY PLOTS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''



def anomalie_cbar_1(vmax, l1,add_white):
    import matplotlib as mpl
    
    
    if len(l1) == 0: # THis means I have set my own custom levels for the proble
        if vmax == 5:
            l1 = np.array([1.5,2,2.5,3,3.5,4, 4.5,5])
        if vmax == 3.2:
            l1 = np.array([1.2,1.4,1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2])
        if  vmax == 3:
            l1 = np.array([1.25,1.5,1.75,2,2.5,3])
        elif vmax == 2:
            l1 = np.array([1.2,1.4,1.6,1.8,2])
        elif vmax == 2.1:
            l1 = np.array([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2, 2.1])
        elif vmax == 1.5:
            l1 = np.array([1.1,1.2,1.3,1.4,1.5])
        elif vmax == 1.1:
            l1 = np.array([1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09,1.1])
    
    # The decimal values are the inverse of these values
    l2 = 1/l1 
    
    # Need to order them in the right direction
    l2 = np.flip(l2) 
    
    # Comining everything back together
    levels = np.concatenate((l2,np.array([1]),l1))
    
    # Creating a colorbar with the levels where you want them
    custom_RdBu = plt.cm.get_cmap("RdBu",len(levels))(np.arange(len(levels)))
    
#     Find the middle of the color bar
    if add_white:
        upper_mid = np.ceil(len(custom_RdBu)/2)
        lower_mid = np.floor(len(custom_RdBu)/2)
        white = [1,1,1,1]

        custom_RdBu[int(upper_mid)] = white
        custom_RdBu[int(lower_mid)] = white
        custom_RdBu[int(lower_mid) - 1] = white

    #     middle = int(np.mean(len(custom_RdBu)/2))
    #     for i in range(4):
    #         custom_RdBu = np.insert(custom_RdBu,middle, [1,1,1,1], axis = 0)

    
    cmap_custom_RdBu = mpl.colors.LinearSegmentedColormap.from_list("RdWtBu", custom_RdBu,len(levels))
    
    return cmap_custom_RdBu, levels


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def anomalie_cbar_2(cax, levels,vmax,pdata, cbar_title):
    
    tick_locations = levels
    if len(tick_locations) > 10: # There are too many ticks, lets get rid of half
        tick_locations = tick_locations[::2]


    tick_strings = np.round(tick_locations,2).astype(str)

    cbar = plt.colorbar(pdata, cax = cax,orientation = 'horizontal',
                        ticks = tick_locations)


    
    cbar.ax.set_xticklabels(tick_strings, fontsize = 15) 
    cbar.ax.set_title(cbar_title,size = 20)
    
    
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''    
    
def anomalies_plots(datafile,vmax = 3, title = '', cbar_title = '',cbar_num_steps = 10, add_white = 0,  savefig = 0 , savedir = '',
                 cmap = plt.cm.Blues, l1 = [], dontplot = 0):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import BoundaryNorm
    import matplotlib.colors as mpc
    import miscellaneous as misc

    fig = plt.figure(figsize = (24,12))
    gs = gridspec.GridSpec(4,3,hspace = 0.5, wspace = 0, height_ratios = [0.2, 1,1,1])
    fig.suptitle(title, fontsize = 35, y = 0.97)
    
    
    phases = np.append(np.arange(1,9),'inactive')
    print(len(l1))
    if len(l1) != 0: # This means I am adding my own levels in that I want, nor a preset levels
        vmax = max(l1)
        
    custom_RdBu, levels = anomalie_cbar_1(vmax, l1, add_white)
    vmin = 1/vmax

    datafile = misc.remove_outside_point(datafile, vmax, vmin)
        
    for i,phase in enumerate(phases):
       
        ax = fig.add_subplot(gs[i + 3], projection  = ccrs.PlateCarree())
    
        data = datafile.sel(phase = str(phase))
        
#         data = data.fillna(1)
#         data = data.where(data > vmin, vmin)
#         data = data.where(data < vmax, vmax)
#         data = data.where(data != 1, np.nan)
        
        
        pdata = data.plot(ax = ax, cmap = custom_RdBu, 
                             vmin = vmin , vmax = vmax,
                             add_colorbar = False,
                             norm = BoundaryNorm(levels, len(levels)-1)) 

        #Removing the spines of the plot. Cartopy requires different method
        ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
        
        if phase == 'inactive':
            ax.set_title('Inactive Phase', size = 25)
        else:
            ax.set_title('Phase ' + str(phase), size = 25)
        
        '''~~~~~ Colorbar'''
    axes = plt.subplot(gs[0,1:2])
    anomalie_cbar_2(axes,levels,vmax, pdata, cbar_title)
    
    
    if savefig:
        fig.savefig(savedir + title + '.png', dpi = 400)
        
    if dontplot:
        plt.fig(close)
    

    
    
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~TREND PLOTS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''




def trend_plots(data, stip_data = '',
                        title = '', colorbar_title = '', 
                        vmax = 40, savefig = 0, savedir = ''):
            
    '''~~~~~~~~~~~~~~~~~
    Dummy file to correct the error with colorf'''
    import matplotlib.colors as mpc
    vmin = -vmax


    '''~~~~~~~~~~~~~~~~~'''
    fig  = plt.figure(figsize = (30, 12))
    gs = gridspec.GridSpec(4,3, hspace = 0.5, wspace = 0, height_ratios = [0.2, 1,1,1])
    fig.suptitle(title, fontsize = 35, y = 0.97)


    # cmap = cmap_test
#     cmap = plt.cm.RdBu
#     levels = np.arange(vmin, vmax, 10)
#     custom_cmap = plt.cm.get_cmap('RdBu', len(levels))(np.arange(len(levels)))
#     cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels))
    num_divisions = 10
    levels = np.linspace(vmin, vmax, num_divisions)
    cmap = plt.cm.get_cmap('RdBu', num_divisions) 
    
    
    row = 0
    column = 0
    all_plot = []
    
    phases = np.append(np.arange(1,9),'inactive')
    for i,phase in enumerate(phases):
       
        data_phase = data.sel(phase = phase)
        ax = fig.add_subplot(gs[i + 3], projection  = ccrs.PlateCarree())
#         plot = data_phase.plot(ax = ax,cmap = cmap, add_colorbar = False, vmax = vmax, vmin = vmin)
        X,Y = np.meshgrid(data_phase.lon, data_phase.lat)
        plot = ax.contourf(X,Y, data_phase.values, cmap = cmap, vmin = vmin, vmax = vmax, levels = levels)

        
        if type(stip_data) != str:
#             sub_sig = stip_data.sel(phase = phase)

#             sig = sub_sig.where(~np.isfinite(sub_sig), 1)
#             size = np.nan_to_num(sig.values, 0)
#             ax.scatter(X,Y, s = size, color = 'k', alpha = 0.8)
            
            stip_sub  = stip_data.sel(phase = phase)
            ax.pcolor(X,Y,stip_sub.values, hatch = '.', alpha = 0, zorder = 100)

        ax.outline_patch.set_visible(False)#Removing the spines of the plot. Cartopy requires different method
        ax.coastlines(resolution = '50m')
        ax.set_title(str(phase).capitalize(), size = 25)

        column += 1

        if column == 4:
            column = 0
            row += 1


    '''~~~~~ Colorbar'''
    axes = plt.subplot(gs[0,1:2])
    # cbar = difference_colorbar_horizontal(axes, plot,vmin, vmax , orientation = 'horizontal')
    cbar = plt.colorbar(plot, cax=axes, orientation = 'horizontal')#,norm = norm)
    cbar.ax.set_title(colorbar_title, size = 20);
    
    ticks = np.linspace(-vmax, vmax, 11)
    cbar.set_ticks(ticks)
    tick_labels = np.core.defchararray.add(ticks.astype(int).astype(str), np.tile('%', len(ticks)))
    cbar.ax.set_xticklabels(tick_labels, fontsize = 15)
#     tick_labels = np.arange(vmin,vmax + 5, 10).astype('str')
#     tick_labels = np.core.defchararray.add(tick_labels  , np.tile('%',len(tick_labels)));
#     cbar.ax.set_xticklabels(tick_labels, fontsize = 15);
    
    
    if savefig:
        fig.savefig(savedir + title + '.png', dpi = 400)
        
        