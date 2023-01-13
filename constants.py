MJO_DATA_PATH='/g/data/w40/ab2313/mjo_and_rainfall_trend'

IMAGE_SAVE_DIR = '/g/data/w40/ab2313/images/mjo_and_rainfall_trend_images'



brown  = '#c38e3f'
green = '#44a198'

from typing import NamedTuple

class NWABounds(NamedTuple):
    lat = (-25, -10)
    lon = (110, 135)