import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask.array
import cartopy.crs as ccrs
import matplotlib.colors as colors
import datetime as dt
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import matplotlib.gridspec as gridspec

import matplotlib.patches as patch
import matplotlib.colors as mpc
import miscellaneous as misc
import load_dataset as load

import sys
sys.path.append('../functions/')

from miscellaneous import apply_masks



def fig_formatter(height_ratios , width_ratios,  hspace = 0.4, wspace = 0.2):
    
    height = np.sum(height_ratios)
    width = np.sum(width_ratios)
    num_rows = len(height_ratios)
    num_cols = len(width_ratios)
    
    fig  = plt.figure(figsize = (10 * width,5 * height)) 
    gs = gridspec.GridSpec(num_rows ,num_cols, hspace = hspace, 
                           wspace = wspace, height_ratios = height_ratios, width_ratios = width_ratios)
    return fig, gs



def colorbar_creater(vmax, step, cmap = 'RdBu', vmin = '', add_white = 0, extender = 0):
    
    # If no vmin value is enetered
    if type(vmin) == str:
        vmin = -vmax
        
    # These are the different bound
    levels = np.arange(vmin, vmax + step, step)
    
    # For this plot, in order for the stippling to be seen, the dark colors at the end need to be clipped off
    # This is doen be extending the cmap further on either side, then clipping the ends off
     # This is the extra amount of discrete colors to make
        # List  of all the colors
    custom_cmap = plt.cm.get_cmap(cmap, len(levels) + extender)(np.arange(len(levels) + extender)) 
    if extender: # Chopping of some colors that are to dark to see the stippling
        custom_cmap = custom_cmap[extender:-extender] # CLipping the ends of either side
#         custom_cmap = custom_cmap[0:-extender]

#         custom_cmap[0] = [1,1,1,1] # first element is white

    # This will add a white section to the middle of the color bar. This is useful is small values 
    # need not be shown.
    if add_white:
        
        # Finding the two middle points. Cbar should be symmertic and will thus have two vales that 
        # are in the middle
        upper_mid = np.ceil(len(custom_cmap)/2)
        lower_mid = np.floor(len(custom_cmap)/2)
        
        # This is the rgb code for white with alpha = 1.
        white = [1,1,1,1]


        custom_cmap[int(upper_mid)] = white
        custom_cmap[int(lower_mid)] = white
        
        # This must also be set to white. Not quite sure of the reasoning behind this. 
        custom_cmap[int(lower_mid) - 1] = white
     
        
    
    # Joing back together as a colormap.
    cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels)) 
    
    return cmap, levels


def lat_lon_grid(ax):

        ax.set_xticks([120,130,140,150] , crs = ccrs.PlateCarree())
        ax.set_xticklabels(['120E','130E','140E','150E'], size = 12)
        ax.set_xlabel('')
        
        
        ax.set_yticks([-12, -16,-20] , crs = ccrs.PlateCarree())
        ax.set_yticklabels(['12S','16S','20S'], size = 12)
        ax.set_ylabel('')

def map_plot_with_stippling_and_NWASquare(data, ax, cmap, levels, square = 0, stip_data = '', stip_reduce = 0, 
                            sig_size = 1, sig_alpha = 0.5, lat_lon = 1, ptype = '',
                           title = ''):
        
        # DESCRIPTION
        # data: Xarray data array with lat and lon coords only.
        # ax: the axis to be plotted on.
        # the colormap to be used.
        # square: this will add a square in the region where there is the trend in the north-west
        #         of Australia.
        # stip_data: data set for stippling showing significance. Same format as data.
        # stip_reduce: reduces the amount of stippling.
        # sig_size: the size of significant points.
        # sig_alpha: the alpha value of significant points. 

        # CODE
        
        if type(ptype) == str:
            ptype = ax.contourf
        
        
        # Creating a grid using the lat and lon points.
        data = apply_masks(data)
        stip_data = apply_masks(stip_data)
        
        X,Y = np.meshgrid(data.lon, data.lat)
        # Plotting  the data with cmap and levels.
        plot = ptype(X,Y, data, cmap = cmap, levels = levels)
        ax.set_title(title, fontsize = 20, pad = 5)
        ax.set_extent([112,153, -22,-10])
        
        

        
        if lat_lon:
            lat_lon_grid(ax)
    
        # This patch marks the square where the raifnall trend occurs
        if square:
            ax.add_patch(patch.Rectangle((113.8,-23),21.2,10.8, fill = False, linestyle = '--', linewidth = 1, 
                                    color = 'k', alpha  = 1)) 
            
        # Hatching out the region in plot.
        # Loading in mask to be used in plot.
        mask = load.load_mask()
        md = mask.sel(lat = slice(-23, -16), lon = slice(122,135)).where(mask == 0).mask
        hatchX,hatchY = np.meshgrid(md.lon.values, md.lat.values)
        ax.pcolor(hatchX,hatchY, md,hatch = 'X', alpha = 0)
        
        # Plotting stippling to indicate where the significant data occurs. 
        if type(stip_data) != str:
      
            X,Y = np.meshgrid(stip_data.lon, stip_data.lat)
            
            # Sig points are 1 and nan values are 0.
            # Where there is values this value will become a 1.
            sig = stip_data.where(~np.isfinite(stip_data), 1)
            # All the nan values become 0.
            size = np.nan_to_num(sig.values, 0)
            
            # Reducing the density of stippling if required. 
            if stip_reduce:
                size[::2] = 0
                size[::5] = 0
                size = np.transpose(size)
                size[::2] = 0
                size[::5] = 0
                size = np.transpose(size)
            
            # Scattering the significant values with points being scaled by sig_size.
            # If the points are sig then they are 1 * sig_size sized, if they are not
            # then they are always 0 in size the scatter values.
            ax.scatter(X,Y, s = size * sig_size, color = 'k', alpha = sig_alpha)

        # Removing the spines of the plot. Cartopy requires different method
#         ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
        # Plot is return for the colorbar.
        return plot


def map_plot_with_no_controurf_NWASquare(data, ax, cmap, levels, square = 0,
                            lat_lon = 1, ptype = '',
                           title = ''):
        
        
        # DESCRIPTION
        # data: Xarray data array with lat and lon coords only.
        # ax: the axis to be plotted on.
        # the colormap to be used.
        # square: this will add a square in the region where there is the trend in the north-west
        #         of Australia.
        # stip_data: data set for stippling showing significance. Same format as data.
        # stip_reduce: reduces the amount of stippling.
        # sig_size: the size of significant points.
        # sig_alpha: the alpha value of significant points. 

        # CODE
        # Creating a grid using the lat and lon points.
        data = apply_masks(data)
        

        plot = data.plot(ax = ax, cmap = cmap, levels = levels, add_colorbar = False)

        ax.set_title(title, fontsize = 20, pad = 5)
        ax.set_extent([112,153, -22,-10])
        
        if lat_lon:
            lat_lon_grid(ax)
    
        # This patch marks the square where the raifnall trend occurs
        if square:
            ax.add_patch(patch.Rectangle((113.8,-23),21.2,10.8, fill = False, linestyle = '--', linewidth = 1, 
                                    color = 'k', alpha  = 1)) 
            
        # Hatching out the region in plot.
        # Loading in mask to be used in plot.
        mask = load.load_mask()
        md = mask.sel(lat = slice(-23, -16), lon = slice(122,135)).where(mask == 0).mask
        hatchX,hatchY = np.meshgrid(md.lon.values, md.lat.values)
        ax.pcolor(hatchX,hatchY, md,hatch = '//', alpha = 0)
   

        # Removing the spines of the plot. Cartopy requires different method
#         ax.outline_patch.set_visible(False)
        ax.coastlines(resolution = '50m')
        
        # Plot is return for the colorbar.
        return plot

def create_colorbar(plot, cax, levels, ticks = '', cbar_title = '', cbar_titleSize = 12, xtickSize = 15, rotation = 45,
                   orientation = 'horizontal', cut_ticks = 1,labelpad=30, title_rotation=0,
                   shrink=1):#, title_pad=15):
    # DESCRIPTIN
    # plot: the plot that th cbar is refering to.
    # caxes: the colorbar axes.
    # levels: the levels on the plot
#     mpl.rcParams['text.usetex'] = False
    # CODE
    cbar = plt.colorbar(plot, cax = cax, orientation = orientation, shrink=shrink)#,norm = norm
    cbar.set_ticks(levels[::cut_ticks])

    # Tick label control
    if type(ticks) == str(): # No custom ticks entered. Ticks are rounded level
        tick_labels = np.round(levels,2)
    else: # we do have custom ticks
        tick_labels = ticks
        
    # Orientation control
    if orientation == 'horizontal':
        cbar.ax.set_xticklabels(tick_labels, fontsize = xtickSize, rotation = rotation)
        cbar.ax.set_title(cbar_title, size = cbar_titleSize)
    else:
        cbar.ax.set_yticklabels(tick_labels, fontsize = xtickSize, rotation = rotation), #labelpad=title_pad)
        cbar.ax.set_ylabel(cbar_title, size = cbar_titleSize, rotation=title_rotation, labelpad = labelpad)
