"""
Utility functions to for atmospheric data wrangling / preparation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import basemap
import xray

import atmos.xrhelper as xr
from atmos.utils import print_if

# ----------------------------------------------------------------------
def set_lons(data, lons, lonmax=360):
    """Set data longitudes to 0-360E or 180W-180E convention.

    Parameters
    ----------
    data : ndarray
        Input data array with longitude as the last dimension
    lons : 1-D ndarray or list
        Longitudes of input data
    lonmax : int, optional
        Maximum longitude for output data.  Set to 360 for 0-360E,
        or set to 180 for 180W-180E.

    Returns
    -------
    data_out, lons_out : ndarray
        The data and longitude arrays shifted to the selected convention.
    """
    lonmin = lonmax - 360
    if lonmin >= lons.min() and lonmin <= lons.max():
        lon0 = lonmin
        start = True
    else:
        lon0 = lonmax
        start = False

    data_out, lons_out = basemap.shiftgrid(lon0, data, lons, start=start)
    return data_out, lons_out


# ----------------------------------------------------------------------
def get_topo(lon, lat, datafile='data/topo/ncep2_ps.nc'):
    """Read surface pressure climatology and interpolate to latlon grid."""

    ds = xr.ncload(datafile)
