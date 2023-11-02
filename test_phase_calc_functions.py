import sys
import os

import numpy as np
import pandas as pd
import xarray as xr
import pytest


# Custom Module Imports
sys.path.append(os.path.join(os.getcwd(), 'Documents', 'mjo_and_rainfall_trend'))
import phase_calc_functions as phase_calc


def test_wet_season_year():
    '''
    The data in Jan, Feb, March is moved backwards by one year. So in this test, all the dates that
    # occur in January should be moved backward by one year. 
    '''
    

    mock_da = xr.DataArray(
            data=[1, 2, 3, 4],
            coords={'time': pd.date_range(start='2021-12-31', periods=4, freq='D')},
            dims=['time']
        )

    # Call the function with the mock data
    result = phase_calc.wet_season_year(mock_da)

    # Define the expected result (manually or based on your knowledge of the function)
    expected_result = xr.DataArray(
        data=[2, 3, 4, 1],
        coords={'time': pd.date_range(start='2021-12-31', periods=4, freq='D')},
        dims=['time']
    )

    # Assert that the actual result matches the expected result
    print(mock_da)
    print(result)
    print(expected_result)
    np.testing.assert_array_equal(result.values, expected_result.values)    



def test_split_into_1to8():

    reps: int = 4
    active_phase_data = np.tile(np.arange(1, 9), reps)
    inactive_phase_data = active_phase_data
    phase_data = np.concatenate([active_phase_data, inactive_phase_data])
    amplitude_data = [5] * len(inactive_phase_data) + [0] * len(inactive_phase_data)

    time_range = pd.date_range(start='1974-01-01', freq='D', periods=len(phase_data))

    # Create the mock xarray dataset
    mock_rmm = xr.Dataset({
        'phase': xr.DataArray(phase_data, dims=['time'], coords={'time': time_range}),
        'amplitude': xr.DataArray(amplitude_data, dims=['time'], coords={'time': time_range})
    })

    mock_awap = xr.DataArray(
                data=np.concatenate([active_phase_data, [0] * len(inactive_phase_data)]),
                coords={'time': time_range},
                dims=['time']
            )
    mock_awap.name = 'precip'

    result = phase_calc.split_into_1to8(mock_awap, mock_rmm)

    assert all(result.sel(phase=1).dropna(dim='time').values == 1)
    assert all(result.sel(phase=2).dropna(dim='time').values == 2)
    assert all(result.sel(phase=3).dropna(dim='time').values == 3)
    assert all(result.sel(phase=4).dropna(dim='time').values == 4)
    assert all(result.sel(phase=5).dropna(dim='time').values == 5)
    assert all(result.sel(phase=6).dropna(dim='time').values == 6)
    assert all(result.sel(phase=7).dropna(dim='time').values == 7)
    assert all(result.sel(phase=8).dropna(dim='time').values == 8)
    assert all(result.sel(phase=0).dropna(dim='time').values == 0)




