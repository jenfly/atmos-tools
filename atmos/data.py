"""
Utility functions to for atmospheric data wrangling / preparation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import basemap
import xray

import atmos.xrhelper as xr
from atmos.utils import print_if, strictly_decreasing

# ----------------------------------------------------------------------
def set_lon(data, lon, lonmax=360):
    """Set data longitudes to 0-360E or 180W-180E convention.

    Parameters
    ----------
    data : ndarray
        Input data array with longitude as the last dimension
    lon : 1-D ndarray or list
        Longitudes of input data
    lonmax : int, optional
        Maximum longitude for output data.  Set to 360 for 0-360E,
        or set to 180 for 180W-180E.

    Returns
    -------
    data_out, lon_out : ndarray
        The data and longitude arrays shifted to the selected convention.
    """
    lonmin = lonmax - 360
    if lonmin >= lon.min() and lonmin <= lon.max():
        lon0 = lonmin
        start = True
    else:
        lon0 = lonmax
        start = False

    data_out, lon_out = basemap.shiftgrid(lon0, data, lon, start=start)
    return data_out, lon_out


# ----------------------------------------------------------------------
def lon_convention(lon):
    """Return 360 if longitudes are 0-360E, 180 if 180W-180E.

    The output of this function can be used in the set_lon() function
    to make two data arrays use a consistent longitude convention.
    """
    if lon.min() < 0:
        return 180
    else:
        return 360


# ----------------------------------------------------------------------
def interp_latlon(data, lat_in, lon_in, lat_out, lon_out, checkbounds=False,
                  masked=False, order=1):
    """Interpolate data array onto a new lat-lon grid.

    Parameters
    ----------
    data : 2-D array
        Data to interpolate, with latitude as first dimension,
        longitude second
    lat_in, lon_in : 1-D float array
        Latitude and longitudes of input data
    lat_out, lon_out : 1-D float array
        Latitude and longitudes to interpolate onto
    checkbounds : bool, optional
        If True, values of lat_out and lon_out are checked to see
        that they lie within the range specified by lat_in, lon_in.
        If False, and lat_out, lon_out are outside lat_in, lon_in,
        interpolated values will be clipped to values on boundary
        of input grid lat_in, lon_in
    masked : bool or float, optional
        If True, points outside the range of lat_in, lon_in are masked
        (in a masked array).
        If masked is set to a number, then points outside the range of
        lat_in, lon_in will be set to that number.
    order : int, optional
        0 for nearest-neighbor interpolation,
        1 for bilinear interpolation
        3 for cublic spline (requires scipy.ndimage).

    Returns
    -------
    data_out : 2-D array
        Data interpolated onto lat_out, lon_out grid
    """

    # Check for the common case that lat_in and/or lat_out are decreasing
    # and flip if necessary to work with basemap.interp()
    flip = False
    if strictly_decreasing(lat_in):
        lat_in = lat_in[::-1]
        data = data[::-1, :]
    if strictly_decreasing(lat_out):
        flip = True
        lat_out = lat_out[::-1]

    x_out, y_out = np.meshgrid(lon_out, lat_out)

    data_out = basemap.interp(data, lon_in, lat_in, x_out, y_out,
                              checkbounds=checkbounds, masked=masked,
                              order=order)
    if flip:
        data_out = data_out[::-1, :]

    return data_out


# ----------------------------------------------------------------------
def get_topo(lon, lat, datafile='data/topo/ncep2_ps.nc'):
    """Read surface pressure climatology and interpolate to latlon grid."""

    ds = xr.ncload(datafile)
