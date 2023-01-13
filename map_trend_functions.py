   
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
import matplotlib.ticker as mticker
rb = plt.cm.RdBu
bm = plt.cm.Blues
best_blue = '#9bc2d5'
recherche_red = '#fbc4aa'
import matplotlib.patches as patch
import os
import load_dataset as load
import miscellaneous#apply_masks
import warnings
warnings.filterwarnings('ignore')
import matplotlib.colors as mpc

import constants

    

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~TREND PLOTS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''




def format_lat_lon(ax):
    ax.coastlines(resolution = '50m')
    ax.set_xticks([120,130,140,150] , crs = ccrs.PlateCarree())
    ax.set_xticklabels(['120E','130E','140E','150E'], size = 15)
    ax.set_extent([109.5,153, -25,-9.5])
    ax.set_xlabel('')
    
    ytick_locks = np.array([-10, -15, -20, -25])
    ax.set_yticks(ytick_locks , crs = ccrs.PlateCarree())
    ax.set_yticklabels([str(abs(s)) + 'S' for s in ytick_locks], size = 15)
    ax.set_ylabel('')
    

    ax.add_patch(patch.Rectangle((110, -25),135-110, 25-10,
                                 fill = False, linestyle = '--', linewidth = 1, 
                                    color = 'k', alpha  = 0.8)) 

    # Hatching the gibson desert region
    
    mask = load.load_mask()
    md = mask.sel(lat = slice(-30, -16), lon = slice(122,135)).where(mask == 0).mask
    hatchX,hatchY = np.meshgrid(md.lon.values, md.lat.values)
    ax.pcolor(hatchX,hatchY, md,hatch = '//', alpha = 0)

    
    
    
def plot_stippled_data(sub_sig, ax, stiple_reduction=1, sig_size=2.5):

    # The result will be scattered, so we need a meshgrid.
    X,Y = np.meshgrid(sub_sig.lon, sub_sig.lat)

    # Nan values are getting replaced by 1.
    #sig = sub_sig.where(~np.isfinite(sub_sig), 1)

    #Non-nan values (finite values) are getting replaced by 1. 
    sig = sub_sig.where(~np.isfinite(sub_sig), 1)

    # All the values that are nan will be replaced with 0. 
    size = np.nan_to_num(sig.values, 0)

    if stiple_reduction:
        size[::2] = 0
        size = np.transpose(size)
        size[::2] = 0
        size = np.transpose(size)
    ax.scatter(X,Y, s = size * sig_size, color = 'grey', alpha = 1)

def trend_plots(data, stip_data = None,
                        vmax = 40, step = 10, sig_size = 2.5, vmin=None, 
                        title = '', colorbar_title = '', 
                        tick_symbol = '%', round_level = 0,
                        savedir = None, cmap = plt.cm.RdBu, stiple_reduction=None):
            

    
    data = miscellaneous.apply_masks(data)

    '''~~~~~~~~~~~~~~~~~ Plot stetup'''
    # Checking if the is phase (9 plots; 3 x 3 grid) of subphase (4 plots; 2 x 2 grid)
    
    # We need to also account for passing in a single phase. If we do then len(data.phase.values) won't work.

    numphase = len(data.phase.values)
    if numphase == 9:
        print('Activated')
        num_rows = 3
        num_cols = 3
        hspace = 0.01
    else:
        num_rows = 2
        num_cols = 2
        hspace = 0.4

    
    fig  = plt.figure(figsize = (3 * 10, num_rows * 20/3)) #20/3 is the height adjust factor b/w subphase and phase
    gs = gridspec.GridSpec(num_rows + 1,num_cols, hspace = hspace, wspace = .2, height_ratios = [0.2] + num_rows * [1])

    '''~~~~~~~~~~~~~~~~~ Creating a custom colorbar'''   
    vmin = -vmax if vmin is None else vmin
    cmap = cmap
    levels = np.arange(vmin, vmax + step, step)
    print(levels)
   
    
    # For this plot, in order for the stippling to be seen, the dark colors at the end need to be clipped off
    # This is doen be extending the cmap further on either side, then clipping the ends off
    extender = 4 # This is the extra amount of discrete colors to make
    custom_cmap = plt.cm.get_cmap(cmap, len(levels) + extender)(np.arange(len(levels) + extender)) # List  of all the colors
    custom_cmap = custom_cmap[2:-2] # CLipping the ends of either side
    cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels)) # Joining the colormap back together
    cmap = 'BrBG'
    
    
    '''~~~~~~~~~~~~~~~~~ Plotting Values'''   
    row = 1 # Starting on the first row, as colorbar goes on the zero row
    column = 0
    
    phases = data.phase.values
    for i,phase in enumerate(phases):
       
        data_phase = data.sel(phase = phase)
        ax = fig.add_subplot(gs[row, column], projection  = ccrs.PlateCarree())
        
        plot = data_phase.plot(ax = ax, cmap = cmap, levels = levels, add_colorbar = False, extend='neither')
        
        
        if isinstance(stip_data, xr.DataArray): plot_stippled_data(stip_data.sel(phase=phase), ax,
                                                                   stiple_reduction=stiple_reduction,
                                                                  sig_size=sig_size)
            
        format_lat_lon(ax)
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
    tick_labels = ticks.round(round_level).astype(str) # .astype(int)
    if isinstance(tick_symbol, str):
        tick_labels = np.core.defchararray.add(tick_labels, np.tile(tick_symbol, len(ticks)))
        
    cbar.ax.set_xticklabels(tick_labels, fontsize = 20)

      
    fig.suptitle(title.replace('_', ' ').title(), y=0.99, fontsize=28)
    
    if savedir is not None:
        save_path = os.path.join(savedir, title + '.png')
        print(f'Saving to {save_path}')
        fig.savefig(save_path, dpi = 400)


def trend_plots_vertical(data, stip_data = '',
                        vmax = 40, step = 10, sig_size = 2.5,
                        title = '', colorbar_title = '', 
                        savedir = '', cmap = plt.cm.RdBu):
            

    import matplotlib.colors as mpc

    '''~~~~~~~~~~~~~~~~~ Plot stetup'''
    # Checking if the is phase (9 plots; 3 x 3 grid) of subphase (4 plots; 2 x 2 grid)
    
    # We need to also account for passing in a single phase. If we do then len(data.phase.values) won't work.

    numphase = len(data.phase.values)
    num_rows = numphase
    num_cols = 1
#     if numphase == 9:
#         num_rows = 3
#         num_cols = 3
#         hspace = 0.1
#     else:
#         num_rows = 2
#         num_cols = 2
    hspace = 0.4

    
    fig  = plt.figure(figsize = (3 * 10, num_rows * 20/3)) #20/3 is the height adjust factor b/w subphase and phase
    gs = gridspec.GridSpec(num_rows + 1,num_cols, hspace = hspace, wspace = 0, height_ratios = [0.2] + num_rows * [1])
    fig.suptitle(title, fontsize = 35, y = 0.97)


    '''~~~~~~~~~~~~~~~~~ Creating a custom colorbar'''   
    vmin = -vmax
    cmap = cmap
    levels = np.arange(vmin, vmax + step, step)
    
    # For this plot, in order for the stippling to be seen, the dark colors at the end need to be clipped off
    # This is doen be extending the cmap further on either side, then clipping the ends off
    extender = 4 # This is the extra amount of discrete colors to make
    custom_cmap = plt.cm.get_cmap(cmap, len(levels) + extender)(np.arange(len(levels) + extender)) # List  of all the colors
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
        
        
        # If there is stip_data: This data is of the form nan where not significant
        # and finite valued where significant
        if type(stip_data) != str:
            
            # Getting just the required phase.
            sub_sig = stip_data.sel(phase = phase)
            
            # The result will be scattered, so we need a meshgrid.
            X,Y = np.meshgrid(sub_sig.lon, sub_sig.lat)
            
            # Nan values are getting replaced by 1.
            #sig = sub_sig.where(~np.isfinite(sub_sig), 1)
            
            #Non-nan values (finite values) are getting replaced by 1. 
            sig = sub_sig.where(~np.isfinite(sub_sig), 1)
            
            # All the values that are nan will be replaced with 0. 
            size = np.nan_to_num(sig.values, 0)
#             size[::2] = 0
# #             size[::5] = 0
#             size = np.transpose(size)
#             size[::2] = 0
#             size[::5] = 0
#             size = np.transpose(size)
            ax.scatter(X,Y, s = size * sig_size, color = 'grey', alpha = 1)
            
#         ax.outline_patch.set_visible(False)#Removing the spines of the plot. Cartopy requires different method
        
        # Adding in ticks for different lats and lons
        format_lat_lon(ax)
        

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
        fig.savefig(savedir + title + '.png', bbox_inches='tight', dpi = 400)

        

def trend_plot_combined(data, stip_data = '',
                        vmax = 40, step = 10, sig_size = 1,
                        title = '', colorbar_title = '', 
                        savedir = ''):
            
    
    # This plot is for plotting two different indices. The two indices are merged into the one data set 
    # before being loaded into this function.
    import matplotlib.colors as mpc

    '''~~~~~~~~~~~~~~~~~ Plot stetup'''
    numphase = len(data.phase.values)
    
    num_rows = 4 # 4 rows for the differnt phases of the MJO.
    num_cols = 2 # Two columns for the different indices that are being used.
    
    fig  = plt.figure(figsize = (3 * 10, num_rows * 20/3)) #20/3 is the height adjust factor b/w subphase and phase
    gs = gridspec.GridSpec(num_rows + 1,num_cols, hspace = 0.12, wspace = 0.1, height_ratios = [0.2] + num_rows * [1])
#     fig.suptitle(title, fontsize = 35, y = 0.99)


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
    
    phases = data.phase.values
    X,Y = np.meshgrid(data.lon, data.lat)
    
    # Loading in mask to be used in plot.
    mask = load.load_mask()
    md = mask.sel(lat = slice(-23, -16), lon = slice(122,135)).where(mask == 0).mask
    hatchX,hatchY = np.meshgrid(md.lon.values, md.lat.values)
    
    
    # Looping through all fo the indices.
    for column, index in enumerate(data):
        data_index  = data[index]
    
        for row, phase in enumerate(phases):

            data_phase = data_index.sel(phase = phase)
            # Row is + 1 as the color bar is going to be on the first row.
            ax = fig.add_subplot(gs[row + 1, column], projection  = ccrs.PlateCarree())


            # Plotting  the data with cmap and levels.
            plot = ax.contourf(X,Y, data_phase, cmap = cmap, levels = levels)
            if isinstance(stip_data, xr.DataArray): plot_stippled_data(stip_data.sel(phase=phase), ax,
                                                                   stiple_reduction=stiple_reduction,
                                                                  sig_size=sig_size)
            
            # If stip data is an array (stip_data is a string unless specific)
#             if type(stip_data) != str:
#                 sub_sig_index = stip_data[index]
#                 sub_sig = sub_sig_index.sel(phase = phase)
#                 X,Y = np.meshgrid(sub_sig.lon, sub_sig.lat)

#                 sig = sub_sig.where(~np.isfinite(sub_sig), 1)
#                 size = np.nan_to_num(sig.values, 0)
#                 ax.scatter(X,Y, s = size * sig_size, color = 'k', alpha = 0.4)
            
            # Adding in ticks for different lats and lons
            format_lat_lon(ax, title=str(phase).capitalize())
            ax.pcolor(hatchX,hatchY, md,hatch = '//', alpha = 0)
            
            
            if column == 0:
                ax.set_ylabel(str(phase).capitalize(), size = 25, labelpad = 10)
                
            if column == 0 and row == 0:
                ax.set_title('Number of Rain Days', size = 25)
            if column == 1 and row == 0:
                ax.set_title('Total Rainfall (mm)', size = 25)

           


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

def trend_plot_combined_better(data, stip_data = '',
                        vmax = 40, step = 10, sig_size = 1,
                        title = '', colorbar_title = '', col_titles = ['Number of Raindays', 'Rainfall'],
                        savedir = '', cmap = plt.cm.RdBu):
            
    
    # This plot is for plotting two different indices. The two indices are merged into the one data set 
    # before being loaded into this function.
    import matplotlib.colors as mpc


    num_rows = len(data.phase.values) # 4 rows for the differnt phases of the MJO.
    num_cols = 2 # Two columns for the different indices that are being used.
    
    fig  = plt.figure(figsize = (3 * 10, num_rows * 20/3)) #20/3 is the height adjust factor b/w subphase and phase
    gs = gridspec.GridSpec(num_rows + 1,num_cols, hspace = 0.15, wspace = 0.1, height_ratios = [0.2] + num_rows * [1])


    vmin = -vmax
    levels = np.arange(vmin, vmax + step, step)
    
    # For this plot, in order for the stippling to be seen, the dark colors at the end need to be clipped off
    # This is doen be extending the cmap further on either side, then clipping the ends off
    extender = 4 # This is the extra amount of discrete colors to make
    custom_cmap = plt.cm.get_cmap(cmap, len(levels) + extender)(np.arange(len(levels) + extender)) # List  of all the colors
    custom_cmap = custom_cmap[2:-2] # CLipping the ends of either side
    cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels)) # Joingi the colormap back together
    
    
    phases = data.phase.values

              
    X,Y = np.meshgrid(data.lon, data.lat)
    
    # Loading in mask to be used in plot.
    mask = load.load_mask()
    md = mask.sel(lat = slice(-23, -16), lon = slice(122,135)).where(mask == 0).mask
    hatchX,hatchY = np.meshgrid(md.lon.values, md.lat.values)
    
    # Looping through all fo the indices.
    for column, index in enumerate(data):
        data_index  = data[index]
    
        for row, phase in enumerate(phases):

            data_phase = data_index.sel(phase = phase)
            # Row is + 1 as the color bar is going to be on the first row.
            ax = fig.add_subplot(gs[row + 1, column], projection  = ccrs.PlateCarree())


            # Plotting  the data with cmap and levels.
            plot = ax.contourf(X,Y, data_phase, cmap = cmap, levels = levels)
#             plot = data_phase.plot(ax = ax, cmap = cmap, levels = levels, add_colorbar = False)
            
            # If stip data is an array (stip_data is a string unless specific)
            if isinstance(stip_data, xr.DataArray): plot_stippled_data(stip_data.sel(phase=phase), ax,
                                                                   stiple_reduction=stiple_reduction,
                                                                  sig_size=sig_size)
#             if type(stip_data) != str:
#                 sub_sig_index = stip_data[index]
#                 sub_sig = sub_sig_index.sel(phase = phase)
#                 sub_sig = sub_sig.where(mask.mask == 1)
#                 Xs,Ys = np.meshgrid(sub_sig.lon, sub_sig.lat)

#                 sig = sub_sig.where(~np.isfinite(sub_sig), 1)
#                 size = np.nan_to_num(sig.values, 0)
#                 ax.scatter(Xs,Ys, s = size * sig_size, color = 'k', alpha = 0.4)

            format_lat_lon(ax)
            ax.pcolor(hatchX,hatchY, md,hatch = '//', alpha = 0)
            
            
            if column == 0:
                ax.set_ylabel(str(phase).capitalize(), size = 25, labelpad = 10)
                
            if column == 0 and row == 0:
                ax.set_title(col_titles[0], size = 25)
            if column == 1 and row == 0:
                ax.set_title(col_titles[1], size = 25)

           


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


def trend_plot_combined_single_phase(data, stip_data = '',
                        vmax = 40, step = 10, sig_size = 1,
                        title = '', colorbar_title = '', 
                        savedir = '', cmap = plt.cm.RdBu, phase = None):
            
    
    # This plot is for plotting two different indices. The two indices are merged into the one data set 
    # before being loaded into this function.
    import matplotlib.colors as mpc


    num_rows = 1
    num_cols = 2 # Two columns for the different indices that are being used.
    
    fig  = plt.figure(figsize = (3 * 10, num_rows * 20/3)) #20/3 is the height adjust factor b/w subphase and phase
    gs = gridspec.GridSpec(num_rows + 1,num_cols, hspace = 0.5, wspace = 0.1, height_ratios = [0.2] + num_rows * [1])
#     fig.suptitle(title, fontsize = 35, y = 0.99)


    '''~~~~~~~~~~~~~~~~~ Creating a custom colorbar'''   
    vmin = -vmax
    levels = np.arange(vmin, vmax + step, step)
    
    # For this plot, in order for the stippling to be seen, the dark colors at the end need to be clipped off
    # This is doen be extending the cmap further on either side, then clipping the ends off
    extender = 4 # This is the extra amount of discrete colors to make
    custom_cmap = plt.cm.get_cmap(cmap, len(levels) + extender)(np.arange(len(levels) + extender)) # List  of all the colors
    custom_cmap = custom_cmap[2:-2] # CLipping the ends of either side
    cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels)) # Joingi the colormap back together
    
    
    '''~~~~~~~~~~~~~~~~~ Plotting Values'''   

        
    X,Y = np.meshgrid(data.lon, data.lat)
    
    # Loading in mask to be used in plot.
    mask = load.load_mask()
    md = mask.sel(lat = slice(-23, -16), lon = slice(122,135)).where(mask == 0).mask
    hatchX,hatchY = np.meshgrid(md.lon.values, md.lat.values)
    
    
    # Looping through all fo the indices.
    for column, index in enumerate(data):
        data_index  = data[index]
        data_phase = data_index
        
        # Row is + 1 as the color bar is going to be on the first row.
        ax = fig.add_subplot(gs[1, column], projection  = ccrs.PlateCarree())

        # Plotting  the data with cmap and levels.
        print(column, index, X.shape, Y.shape, data_phase.shape)
        plot = ax.contourf(X,Y, data_phase, cmap = cmap, levels = levels)
#             plot = data_phase.plot(ax = ax, cmap = cmap, levels = levels, add_colorbar = False)

        # If stip data is an array (stip_data is a string unless specific)
        if type(stip_data) != str:
            sub_sig_index = stip_data[index]
            sub_sig = sub_sig_index
            Xs,Ys = np.meshgrid(sub_sig.lon, sub_sig.lat)

            sig = sub_sig.where(~np.isfinite(sub_sig), 1)
            size = np.nan_to_num(sig.values, 0)
            ax.scatter(Xs,Ys, s = size * sig_size, color = 'k', alpha = 0.4)

        # ax.outline_patch.set_visible(False)#Removing the spines of the plot. Cartopy requires different method

        # Adding in ticks for different lats and lons
#         ax.coastlines(resolution = '50m')
        ax.annotate(chr(65 + column).lower() + ')', xy = (0.01,1.05), xycoords = 'axes fraction', size = 20)
        format_lat_lon(ax)
        
        
#         ax.set_xticks([120,130,140,150] , crs = ccrs.PlateCarree())
#         ax.set_xticklabels(['120E','130E','140E','150E'], size = 18)
#         ax.set_xlabel('')
#         ax.set_yticks([-12, -16,-20] , crs = ccrs.PlateCarree())
#         ax.set_yticklabels(['12S','16S','20S'], size = 18)
#         ax.set_ylabel('')
#         ax.set_extent([110,153, -25,-10])

#         # This is the square where the rainfall trend is occuring.
#         # (x,y), width, hieght
#         ax.add_patch(patch.Rectangle((113.8,-23),21.2,10.8, fill = False, linestyle = '--', linewidth = 1, 
#                                 color = 'k', alpha  = 1)) 

        # Hatching out the region in plot.
        ax.pcolor(hatchX,hatchY, md,hatch = '//', alpha = 0)



        if column == 0:
            ax.set_title('Number of Rain Days', size = 25)
        if column == 1:
            ax.set_title('Rainfall', size = 25)

           


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

def trend_plot_single(ax, data, cmap, levels, stip_data = None):
            
   
    
    '''~~~~~~~~~~~~~~~~~ Plotting Values'''   

        
    X,Y = np.meshgrid(data.lon, data.lat)
    
    # Loading in mask to be used in plot.
    mask = load.load_mask()
    md = mask.sel(lat = slice(-23, -16), lon = slice(122,135)).where(mask == 0).mask
    hatchX, hatchY = np.meshgrid(md.lon.values, md.lat.values)
    


    # Plotting  the data with cmap and levels.
    plot = ax.contourf(X,Y, data, cmap = cmap, levels = levels)
#             plot = data_phase.plot(ax = ax, cmap = cmap, levels = levels, add_colorbar = False)

    # If stip data is an array (stip_data is a string unless specific)
    if isinstance(stip_data,xr.DataArray):
        
        stip_data = stip_data.where(mask.mask)
        sub_sig = stip_data
        
        sub_sig = sub_sig.where(mask.mask)
        X,Y = np.meshgrid(sub_sig.lon, sub_sig.lat)

        sig = sub_sig.where(~np.isfinite(sub_sig), 1)
        size = np.nan_to_num(sig.values, 0)
        sig_size = 1
        ax.scatter(X,Y, s = size * sig_size, color = 'k', alpha = 0.4)


    # Adding in ticks for different lats and lons
    format_lat_lon(ax)

    ax.pcolor(hatchX,hatchY, md,hatch = '//', alpha = 0)
    
    return plot

          
