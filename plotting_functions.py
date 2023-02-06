import xarray as xr
import numpy as np
import pandas as pd
import dask.array
import cartopy.crs as ccrs
import datetime as dt
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patch
import matplotlib.colors as mpc


import miscellaneous as misc
import load_dataset as load

import sys
import constants
from miscellaneous import apply_masks
import calculation_functions
import mystats


def create_levels(vmax:float, vmin:float=None, step:float=1)->np.ndarray:
    '''
    Ensures that all instances of creating levels using vmax + step as the max.
    '''
    vmin = -vmax if vmin is None else vmin
    return np.arange(vmin, vmax + step, step)

def add_figure_label(ax: plt.Axes, label: str):
    ax.annotate(label, xy = (0.01,1.05), xycoords = 'axes fraction', size=constants.annotate_size)

def format_axis(ax: plt.Axes):
    '''Formatting with no top and right axis spines and correct tick size.'''
    ax.tick_params(axis='x', labelsize=constants.ticklabel_size)
    ax.tick_params(axis='y', labelsize=constants.ticklabel_size)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    
def fig_formatter(height_ratios: List[float] , width_ratios: List[float],  hspace:float = 0.4, wspace:float = 0.2):
    
    height = np.sum(height_ratios)
    width = np.sum(width_ratios)
    num_rows = len(height_ratios)
    num_cols = len(width_ratios)
    
    fig  = plt.figure(figsize = (10*width, 5*height)) 
    gs = gridspec.GridSpec(num_rows ,num_cols, hspace=hspace, 
                           wspace=wspace, height_ratios=height_ratios, width_ratios=width_ratios)
    return fig, gs


def create_discrete_cmap(cmap, number_divisions:int=None, levels=None, vmax=None, vmin=None, step=1,
                         add_white:bool=False, clip_ends:int=0):
    '''
    Creates a discrete color map of cmap with number_divisions
    '''
    
    if levels is not None:
        number_divisions = len(levels)
    elif vmax is not None:
        number_divisions = len(create_levels(vmax, vmin, step))
                
    color_array = plt.cm.get_cmap(cmap, number_divisions+clip_ends)(np.arange(number_divisions+clip_ends)) 
    
    if clip_ends:
        color_array = color_array[clip_ends:-clip_ends]

    if add_white:
        upper_mid = np.ceil(len(color_array)/2)
        lower_mid = np.floor(len(color_array)/2)
        
        white = [1,1,1,1]

        color_array[int(upper_mid)] = white
        color_array[int(lower_mid)] = white
        
        # This must also be set to white. Not quite sure of the reasoning behind this. 
        color_array[int(lower_mid) - 1] = white

    cmap = mpl.colors.ListedColormap(color_array)
    
    return cmap

def colorbar_creater(vmax, step, cmap='RdBu', vmin=None, add_white=False, extender=0):
    print(
        'plotting_functions.colorbar_creater funciton is deprecated.'
        'Please use plotting_functions.create_discrete_cmap instead.')
    
    if vmin is None:
        vmin = -vmax
        
    levels = np.arange(vmin, vmax + step, step)
    
    color_array = plt.cm.get_cmap(cmap, len(levels) + extender)(np.arange(len(levels) + extender)) 
    
    if extender: # Chopping of some colors that are to dark to see the stippling
        color_array = color_array[extender:-extender] # CLipping the ends of either side

    # This will add a white section to the middle of the color bar. This is useful is small values 
    # need not be shown.
    if add_white:
        
        # Finding the two middle points. Cbar should be symmertic and will thus have two vales that 
        # are in the middle
        upper_mid = np.ceil(len(color_array)/2)
        lower_mid = np.floor(len(color_array)/2)
        
        # This is the rgb code for white with alpha = 1.
        white = [1,1,1,1]

        color_array[int(upper_mid)] = white
        color_array[int(lower_mid)] = white
        
        # This must also be set to white. Not quite sure of the reasoning behind this. 
        color_array[int(lower_mid) - 1] = white

    # Joing back together as a colormap.
    cmap = mpl.colors.ListedColormap(color_array)
    #cmap = mpc.LinearSegmentedColormap.from_list("RdWtBu", custom_cmap,len(levels)) 
    
    return cmap, levels

def create_colorbar(plot, cax, levels, ticks='', cbar_title='', cbar_titleSize=constants.cbar_title_size, 
                    xtickSize=constants.ticklabel_size, rotation=45, orientation='horizontal', 
                    cut_ticks=1,labelpad=30, title_rotation=0, shrink=1, round_level=2):
    '''
    plot: the plot that th cbar is refering to.
    caxes: the colorbar axes.
    levels: the levels on the plot
    '''
    
    cbar = plt.colorbar(plot, cax=cax, orientation=orientation, shrink=shrink)
    cbar.set_ticks(levels[::cut_ticks])
    
    tick_labels = np.round(levels,round_level) if isinstance(ticks, str) else ticks
        
    if orientation == 'horizontal':
        cbar.ax.set_xticklabels(tick_labels, fontsize = xtickSize, rotation=rotation)
        cbar.ax.set_title(cbar_title, size=cbar_titleSize)
    else:
        cbar.ax.set_yticklabels(tick_labels, fontsize = xtickSize, rotation=rotation)
        cbar.ax.set_ylabel(cbar_title, size=cbar_titleSize, rotation=title_rotation, labelpad=labelpad)

def format_lat_lon(ax):
    ax.coastlines(resolution = '50m')
    ax.set_xticks([120,130,140,150] , crs = ccrs.PlateCarree())
    ax.set_xticklabels(['120E','130E','140E','150E'], size = constants.ticklabel_size)
    ax.set_extent([109.5,153, -25,-9.5])
    ax.set_xlabel('')
    
    ytick_locks = np.array([-10, -15, -20, -25])
    ax.set_yticks(ytick_locks , crs = ccrs.PlateCarree())
    ax.set_yticklabels([str(abs(s)) + 'S' for s in ytick_locks], size = constants.ticklabel_size)
    ax.set_ylabel('')
    

    ax.add_patch(patch.Rectangle((110, -25),135-110, 25-10,
                                 fill = False, linestyle = '--', linewidth = 1, 
                                    color = 'k', alpha  = 0.8)) 

    # Hatching the gibson desert region
    
    mask = load.load_mask()
    md = mask.sel(lat = slice(-30, -16), lon = slice(122,135)).where(mask == 0).mask
    hatchX,hatchY = np.meshgrid(md.lon.values, md.lat.values)
    ax.pcolor(hatchX,hatchY, md,hatch = '//', alpha = 0)

    
def plot_stippled_data(sub_sig, ax, stiple_reduction=1, sig_size:float=constants.sig_size):

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
    ax.scatter(X,Y, s=size * sig_size, color='grey', alpha=1)



def nwa_map_plot(data: xr.DataArray, ax, cmap, levels: np.ndarray, stip_data: xr.DataArray = None, stip_reduce:int = 0, 
                            sig_size:float=constants.sig_size, sig_alpha:float = 0.5, stiple_reduction=None,
                           title:str = '', norm=None, add_colorbar:bool=False, debug=False):
        '''
        data: Xarray data array with lat and lon coords only.
        ax: the axis to be plotted on.
        the colormap to be used.
        square: this will add a square in the region where there is the trend in the north-west
                of Australia.
        stip_data: data set for stippling showing significance. Same format as data.
        stip_reduce: reduces the amount of stippling.
        sig_size: the size of significant points.
        sig_alpha: the alpha value of significant points. 
        '''

        # Creating a grid using the lat and lon points.
        max_value = np.max(np.abs(levels))
        if debug: print(f'{max_value=}\n{levels=}')
        data = calculation_functions.max_filter(data, max_value)
        data = apply_masks(data)
        plot = data.plot.contourf(ax=ax, cmap=cmap, levels=levels, add_colorbar= add_colorbar, extend='neither',
                                 norm=norm)

        if isinstance(stip_data, xr.DataArray): plot_stippled_data(apply_masks(stip_data), ax,
                                                                   stiple_reduction=stiple_reduction,
                                                                  sig_size=sig_size)

        format_lat_lon(ax)
        ax.set_title(title, size=constants.title_size)
                  
        return plot

def map_plot_with_stippling_and_NWASquare(*args, **kwargs):
    print('This function is obsolete and is currently just wrapping nwa_map_plot')
    return nwa_map_plot(*args, *kwargs)
    

    
def format_cbar_tick_string(levels, round_level, tick_symbol):
    '''
    Ticks may be float and needed to be rounded to round_level decimal places.
    A symbol might also want to be added (e.g. percent sign).
    '''
    ticks = tick_labels = levels.round(round_level).astype(str)
    if isinstance(tick_symbol, str): ticks = np.core.defchararray.add(ticks, np.tile(tick_symbol, len(ticks)))
        
    return ticks
      
# Previosly trend_plot_combined_better
def datavars_as_col_plot(data:xr.DataArray, row_varialbe:str, stip_data=None, vmax=40, vmin=None, step=10, 
                         vmax2=None, vmin2=None, step2=None, hspace=0.25, 
                         sig_size:float=constants.sig_size, stiple_reduction=None, colorbar_title=None,
                         colorbar_title2=None, col_titles=None, 
                         cmap='BrBG', tick_symbol='%', round_level=0,):
                
    row_values = data[row_varialbe].values
    data_vars = list(data.data_vars)
    
    num_rows = len(row_values) 
    num_cols = len(data_vars)
    
    fig, gs = fig_formatter(height_ratios=[0.2] + num_rows * [1], width_ratios=[1, 1], hspace=hspace, wspace=.1)
    
    levels = create_levels(vmax, vmin, step)
    
    # Note: will have to sort something out if adding a thrid colum with third unique colorbar
    levels2 = create_levels(vmax2, vmin2, step2) if vmax2 is not None else levels
        
    level_list = [levels, levels2]

    data = calculation_functions.max_filter(data, np.max([np.max(levels), np.max(levels2)]))

    if col_titles is None: col_titles = data_vars
    
    plot_num = 0
    plot_list = []
    for row, phase in enumerate(row_values):
        for column, index in enumerate(data_vars):    
            stip_data_sel = stip_data[index].sel(phase=phase) if stip_data is not None else None
            ax = fig.add_subplot(gs[row+1, column], projection=ccrs.PlateCarree())
            plot = nwa_map_plot(
                calculation_functions.max_filter(data[index].sel(phase=phase), np.max(level_list[column])), 
                ax, stip_data=stip_data_sel, sig_size=sig_size, cmap=cmap, levels=level_list[column])
            add_figure_label(ax, f'{chr(97+plot_num)})')
            plot_num += 1
            
            if column == 0: ax.set_ylabel(str(phase).capitalize(), size=constants.title_size, labelpad=10)    
            if column == 0 and row == 0: ax.set_title(col_titles[0], size=constants.title_size)
            if column == 1 and row == 0: ax.set_title(col_titles[1], size=constants.title_size)
            plot_list.append(plot)# Only store one plot for every column
        
    if vmax2 is not None:
        cax2 = plt.subplot(gs[0,1])
        ticks2 = format_cbar_tick_string(levels2, round_level, tick_symbol)

        create_colorbar(plot_list[1], cax2, levels2, ticks=ticks2, cbar_title=colorbar_title2,
                       xtickSize=constants.ticklabel_size+3, cbar_titleSize=constants.cbar_title_size+5, 
                       orientation='horizontal', rotation=0)
        cbar1_extent = 1
    
    else:
        cbar1_extent=num_cols
        
    cax = plt.subplot(gs[0,:cbar1_extent])
    ticks = format_cbar_tick_string(levels, round_level, tick_symbol)

    create_colorbar(plot_list[0], cax, levels, ticks=ticks, cbar_title=colorbar_title,
                       xtickSize=constants.ticklabel_size+3, cbar_titleSize=constants.cbar_title_size+5, 
                       orientation='horizontal', rotation=0)
    
    return fig




def all_phase_trend_plots(data: xr.DataArray, stip_data = None,
                        vmax=40, step=10, sig_size:float=constants.sig_size, vmin=None,  title='', colorbar_title='', 
                        tick_symbol='%', round_level=0, cmap = 'BrBG', stiple_reduction=None,
                         return_all_fig_comps:bool=False):
            
    phases = data.phase.values
    numphase = len(phases)
    
    if numphase == 9:
        num_rows = 3
        num_cols = 3
        hspace = 0.55
        wspace=.1
    else:
        num_rows = 2
        num_cols = 2
        hspace = 0.4
        wspace=.08

    
    fig, gs = fig_formatter(height_ratios= [0.2]+num_rows*[1], width_ratios=[1]*num_cols, hspace=hspace, wspace=wspace)
    
    levels = create_levels(vmax, vmin, step)
    data = apply_masks(data)

    row = 1 # Starting on the first row, as colorbar goes on the zero row
    column = 0
    for i,phase in enumerate(phases):
       
        ax = fig.add_subplot(gs[row, column], projection  = ccrs.PlateCarree())
        stip_data_sel = stip_data.sel(phase = phase) if stip_data is not None else None

        plot = nwa_map_plot(data.sel(phase = phase), ax,stip_data=stip_data_sel,
                               sig_size=sig_size, cmap=cmap, levels=levels)

        ax.set_title(str(phase).capitalize() if phase!= 0 else 'Inactive', size=constants.title_size)
        
        add_figure_label(ax, f'{chr(97+i)})')

        column += 1
        if column == num_cols: # we have reched the final column
            column = 0 # Go back to the first column 
            row += 1 # But go to the next row

    cax = plt.subplot(gs[0,:num_cols])
    ticks = format_cbar_tick_string(levels, round_level, tick_symbol)
    create_colorbar(plot, cax, levels, ticks=ticks, cbar_title=colorbar_title,
                       xtickSize=constants.ticklabel_size+2, cbar_titleSize=constants.cbar_title_size+4, 
                       orientation='horizontal', rotation=0)
    
    if return_all_fig_comps:
        # Use fig.get_axes() to get the axes.
        return [fig, gs, cax]
    return fig


def plot_fraction(fig, gs, row_number, ds1, ds2, ds_numerator, ds_denominator,
                  levels_percent, cmap_percent, levels_frac, cmap_frac,
                  row_label=None, col_labels=None,
                  ds1_stip=None, ds2_stip=None, sig_size=1, debug=False):
    '''
    Plots two datasets and then plots the ration of the two dataset as a row.
    Needs to take the gs and row_nmber.
    '''
    
    ax1 = fig.add_subplot(gs[row_number,0], projection=ccrs.PlateCarree())
    c_percent = nwa_map_plot(ds1, ax1, stip_data=ds1_stip, levels=levels_percent, cmap=cmap_percent,
                                sig_size=sig_size, debug=debug)

    ax2 = fig.add_subplot(gs[row_number,1], projection=ccrs.PlateCarree())
    c_percent = nwa_map_plot(ds2, ax2, stip_data=ds2_stip,
                                levels=levels_percent, cmap=cmap_percent, sig_size=sig_size, debug=debug)


    ax3 = fig.add_subplot(gs[row_number,2], projection=ccrs.PlateCarree())
    c_frac = nwa_map_plot(ds_numerator/ ds_denominator, ax3,
                        levels=levels_frac, cmap=cmap_frac, debug=True)
    
    
    ax1.set_ylabel(row_label, fontsize=constants.title_size, labelpad=40)
    
    if col_labels is not None:
        for label, ax in zip(col_labels, [ax1,ax2,ax3]):
            ax.set_title(label, fontsize=constants.title_size)
    
    return c_percent, c_frac

def mjo_single_phase_line_plot(ax, data:xr.DataArray, color=constants.green, trendline_color='Blue', **kwargs):
    
    func_kwargs = dict(base_label='', linecolor = 'blue')
    func_kwargs = {**func_kwargs, **kwargs}
         
    x = data.year.values if 'year' in list(data.dims) else data.time.values
    y = data.values
    z = np.polyfit(x,y,1)
    p = np.poly1d(z)
    
    # Statistical significance
    sig = mystats.mann_kendall(y)
    
    # The trend / mean * 100 (convert to percent) * 10 convert to per decade
    mean = np.mean(y[np.isfinite(y)])
    percent_trend =  np.round(z[0] * 1000/ mean,1)
 
    label = str(np.round(z[0] * 10,1)) + ' days/decade\n({}%/decade)'.format(percent_trend) 
    trend_kwargs = dict(linestyle = '--',color=trendline_color, label=func_kwargs['base_label']+label)
    
    trend = ax.plot(x, p(x), **trend_kwargs) 
    raw = ax.plot(x,y, color=color)

    
def phase_bar_plot(data: xr.DataArray, ylabel:str=None, annotate_position_dict:dict=None,
                  colors:List[str]=None):
    
     
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    x_labels = data.phase.values.astype(str)
    x_labels[-1] = 'Inactive'

    ax.bar(x_labels, data.values, color=colors)

    ax.set_xlabel('MJO RMM Phase', size=constants.cbar_title_size)
    ax.set_ylabel(ylabel, size=constants.cbar_title_size,
                 rotation=0, labelpad=75)

    format_axis(ax)
    ax.axhline([0], color='k', linestyle='--', zorder=-1000, alpha=0.4, linewidth=0.9)
    
    for text, info in annotate_position_dict.items():
    
        ax.annotate(text, xy=info['xy'], ha='center', va='center', size=constants.ticklabel_size, 
                   color=info['color'], annotation_clip=True, zorder=100)

    return fig, ax