MJO_DATA_PATH='/g/data/w40/ab2313/mjo_and_rainfall_trend'

IMAGE_SAVE_DIR = '/g/data/w40/ab2313/images/mjo_and_rainfall_trend_images'


START_MONSOON_MONTHS = [12]
END_MONSOON_MONTHS  = [1,2,3]

# Titles
title_size = 22

# Annotation
annotate_size = 18

# Cbar title, axis labels, legend 
cbar_title_size = 15

# Tick labels on line plots and tick labels for lat and lon on maps
ticklabel_size = 12


sig_size=1

brown  = '#c38e3f'
brown2 = '#a4681b'
green = '#44a198'
best_blue = '#9bc2d5'
recherche_red = '#fbc4aa'
wondeful_white = '#f8f8f7'
from typing import NamedTuple

class NWABounds(NamedTuple):
    lat = (-25, -10)
    lon = (110, 135)
    
    
nw_slice = dict(lat=slice(*NWABounds.lat), lon=slice(*NWABounds.lon))