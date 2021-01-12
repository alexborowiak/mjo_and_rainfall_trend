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
sys.path.append('/home/563/ab2313/MJO/functions')
import subphase_calc_functions as subphase_calc
import mystats



def trend_plots(data, stip = 0,
                        titlepiece = '', datasource = 'AWAP', colorbar_title = '', 
                        vmax = 40, savedir = ''):
            
    '''~~~~~~~~~~~~~~~~~
    Dummy file to correct the error with colorf'''
    import matplotlib.colors as mpc
    vmin = -vmax


    '''~~~~~~~~~~~~~~~~~'''
    fig  = plt.figure(figsize = (6, 12))
    gs = gridspec.GridSpec(5,1, hspace = 0.5, wspace = 0, height_ratios = [0.2,1, 1,1,1])
    title = f'Trend in {titlepiece} for Phases of the MJO in {datasource}'
    fig.suptitle(title, fontsize = 20, y = 0.95)


    # cmap = cmap_test
#     cmap = plt.cm.RdBu
#     levels = np.arange(vmin, vmax, 10)
#     custom_cmap = plt.cm.get_cmap('RdBu', len(levels))(np.arange(len(levels)))
#     cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels))
    num_divisions = 10
    cmap = plt.cm.get_cmap('RdBu', num_divisions) 
    
    
    row = 0
    column = 0
    all_plot = []
    
    phases = data.phase.values
    for i,phase in enumerate(phases):
       
        data_phase = data.sel(phase = phase)
        ax = fig.add_subplot(gs[i + 1], projection  = ccrs.PlateCarree())
        plot = data_phase.plot(ax = ax,cmap = cmap, add_colorbar = False, vmax = vmax, vmin = vmin)

        if stip:
            X, Y = np.meshgrid(data_phase.lon, data_phase.lat)
            ax.pcolor(X,Y, data_phase.values, hatch = '.')
        
        ax.outline_patch.set_visible(False)#Removing the spines of the plot. Cartopy requires different method
        ax.coastlines(resolution = '50m')
        ax.set_title(str(phase).capitalize(), size = 15)

        column += 1

        if column == 4:
            column = 0
            row += 1


    '''~~~~~ Colorbar'''
    axes = plt.subplot(gs[0,:])
    # cbar = difference_colorbar_horizontal(axes, plot,vmin, vmax , orientation = 'horizontal')
    cbar = plt.colorbar(plot, cax=axes, orientation = 'horizontal')#,norm = norm)
    cbar.ax.set_title(colorbar_title, size = 15);
    
    ticks = np.linspace(-vmax, vmax, 11)
    cbar.set_ticks(ticks)
    tick_labels = np.core.defchararray.add(ticks.astype(int).astype(str), np.tile('%', len(ticks)))
    cbar.ax.set_xticklabels(tick_labels, fontsize = 10)
#     tick_labels = np.arange(vmin,vmax + 5, 10).astype('str')
#     tick_labels = np.core.defchararray.add(tick_labels  , np.tile('%',len(tick_labels)));
#     cbar.ax.set_xticklabels(tick_labels, fontsize = 15);
    
    
    if savedir != '':
        fig.savefig(savedir + title + '.png', dpi = 400)