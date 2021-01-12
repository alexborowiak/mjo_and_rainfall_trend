   
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.array
import cartopy.crs as ccrs
import matplotlib.colors as colors
import datetime as dt
from matplotlib.colors import BoundaryNorm
import matplotlib.gridspec as gridspec

rb = plt.cm.RdBu
bm = plt.cm.Blues
best_blue = '#9bc2d5'
recherche_red = '#fbc4aa'


import warnings
warnings.filterwarnings('ignore')


    
    
    
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~TREND PLOTS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''




def trend_plots(data, stip_data = '',
                        vmax = 40, step = 10, sig_size = 2.5,
                        title = '', colorbar_title = '', 
                        savedir = ''):
            

    import matplotlib.colors as mpc

    '''~~~~~~~~~~~~~~~~~ Plot stetup'''
    # Checking if the is phase (9 plots; 3 x 3 grid) of subphase (4 plots; 2 x 2 grid)
    numphase = len(data.phase.values)
    if numphase == 9:
        num_rows = 3
        num_cols = 3
        hspace = 0.1
    else:
        num_rows = 2
        num_cols = 2
        hspace = 0.4
    
    fig  = plt.figure(figsize = (3 * 10, num_rows * 20/3)) #20/3 is the height adjust factor b/w subphase and phase
    gs = gridspec.GridSpec(num_rows + 1,num_cols, hspace = hspace, wspace = 0, height_ratios = [0.2] + num_rows * [1])
    fig.suptitle(title, fontsize = 35, y = 0.97)


    '''~~~~~~~~~~~~~~~~~ Creating a custom colorbar'''   
    vmin = -vmax
    cmap = plt.cm.RdBu
    levels = np.arange(vmin, vmax + step, step)
    
    # For this plot, in order for the stippling to be seen, the dark colors at the end need to be clipped off
    # This is doen be extending the cmap further on either side, then clipping the ends off
    extender = 4 # This is the extra amount of discrete colors to make
    custom_cmap = plt.cm.get_cmap('RdBu', len(levels) + extender)(np.arange(len(levels) + extender)) # List  of all the colors
    custom_cmap = custom_cmap[2:-2] # CLipping the ends of either side
    cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels)) # Joingi the colormap back together
    
    
    '''~~~~~~~~~~~~~~~~~ Plotting Values'''   
    row = 1 # Starting on the first row, as colorbar goes on the zero row
    column = 0
    
    phases = data.phase.values
    for i,phase in enumerate(phases):
       
        data_phase = data.sel(phase = phase)
        ax = fig.add_subplot(gs[row, column], projection  = ccrs.PlateCarree())
        
        plot = data_phase.plot(ax = ax, cmap = cmap, levels = levels, add_colorbar = False)
        
        if type(stip_data) != str:
            sub_sig = stip_data.sel(phase = phase)
            X,Y = np.meshgrid(sub_sig.lon, sub_sig.lat)

            sig = sub_sig.where(~np.isfinite(sub_sig), 1)
            size = np.nan_to_num(sig.values, 0)
            size[::2] = 0
            size[::5] = 0
            size = np.transpose(size)
            size[::2] = 0
            size[::5] = 0
            size = np.transpose(size)
            ax.scatter(X,Y, s = size * sig_size, color = 'k', alpha = 1)
            
            

        ax.outline_patch.set_visible(False)#Removing the spines of the plot. Cartopy requires different method
        ax.coastlines(resolution = '50m')
        ax.set_title(str(phase).capitalize(), size = 25)

        column += 1

        if column == num_cols: # we have reched the final column
            column = 0 # Go back to the first column 
            row += 1 # But go to the next column


    '''~~~~~ Colorbar'''
    axes = plt.subplot(gs[0,:num_cols])
    # cbar = difference_colorbar_horizontal(axes, plot,vmin, vmax , orientation = 'horizontal')
    cbar = plt.colorbar(plot, cax=axes, orientation = 'horizontal')#,norm = norm)
    cbar.ax.set_title(colorbar_title, size = 25);
    
    ticks = levels
    cbar.set_ticks(ticks)
    tick_labels = np.core.defchararray.add(ticks.astype(int).astype(str), np.tile('%', len(ticks)))
    cbar.ax.set_xticklabels(tick_labels, fontsize = 20)
#     tick_labels = np.arange(vmin,vmax + 5, 10).astype('str')
#     tick_labels = np.core.defchararray.add(tick_labels  , np.tile('%',len(tick_labels)));
#     cbar.ax.set_xticklabels(tick_labels, fontsize = 15);
    
    
    if savedir != '':
        fig.savefig(savedir + title + '.png', dpi = 400)
        
        
        
        
        
        
        
        
        
          
