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
def set_lon(data, lonmax=360, lon=None):
    """Set data longitudes to 0-360E or 180W-180E convention.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Input data array with longitude as the last dimension
    lonmax : int, optional
        Maximum longitude for output data.  Set to 360 for 0-360E,
        or set to 180 for 180W-180E.
    lon : 1-D ndarray or list, optional
        Longitudes of input data. Only used if data is an ndarray.
        If data is an xray.DataArray, then lon = data['lon']

    Returns
    -------
    If argument data is an ndarray:
        data_out, lon_out : ndarray
            The data and longitude arrays shifted to the selected
            convention.
    If argument data is an xray.DataArray:
        data_out : xray.DataArray
            DataArray object with data and longitude values shifted to
            the selected convention.
    """

    if isinstance(data, xray.DataArray):
        lon = data['lon']
        vals = data.values
    else:
        vals = data

    lonmin = lonmax - 360
    if lonmin >= lon.min() and lonmin <= lon.max():
        lon0 = lonmin
        start = True
    else:
        lon0 = lonmax
        start = False

    vals_out, lon_out = basemap.shiftgrid(lon0, vals, lon, start=start)

    if isinstance(data, xray.DataArray):
        data_out = data.copy()
        data_out['lon'].values = lon_out
        data_out.values = vals_out
        return data_out
    else:
        return vals_out, lon_out


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
def interp_latlon(data, lat_out, lon_out, lat_in=None, lon_in=None,
                  checkbounds=False, masked=False, order=1):
    """Interpolate data array onto a new lat-lon grid.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Data to interpolate, with latitude as first dimension,
        longitude second
    lat_out, lon_out : 1-D float array
        Latitude and longitudes to interpolate onto.
    lat_in, lon_in : ndarray, optional
        Latitude and longitude arrays of input data.  Only used if data
        is an ndarray. If data is an xray.DataArray then
        lat_in = data['lat'] and lon_in = data['lon']
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
    data_out : 2-D array or xray.DataArray
        Data interpolated onto lat_out, lon_out grid
    """

    if isinstance(data, xray.DataArray):
        lat_in, lon_in = data['lat'].values, data['lon'].values
        vals = data.values
    else:
        vals = data

    # Check for the common case that lat_in and/or lat_out are decreasing
    # and flip if necessary to work with basemap.interp()
    flip = False
    if strictly_decreasing(lat_in):
        lat_in = lat_in[::-1]
        vals = vals[::-1, :]
    if strictly_decreasing(lat_out):
        flip = True
        lat_out = lat_out[::-1]

    x_out, y_out = np.meshgrid(lon_out, lat_out)

    vals_out = basemap.interp(vals, lon_in, lat_in, x_out, y_out,
                              checkbounds=checkbounds, masked=masked,
                              order=order)
    if flip:
        # Flip everything back to previous order
        vals_out = vals_out[::-1, :]
        lat_out = lat_out[::-1]

    if isinstance(data, xray.DataArray):
        data_out = xray.DataArray(vals_out, name=data.name,
                                  coords=[('lat', lat_out), ('lon', lon_out)])
    else:
        data_out = vals_out

    return data_out


# ----------------------------------------------------------------------
def get_topo(lat, lon, datafile='data/topo/ncep2_ps.nc'):
    """Return surface pressure climatology on selected latlon grid."""

    ds = xr.ncload(datafile)
    ps = ds['ps']

    # Check what longitude convention is used in the surface pressure
    # climatology and switch if necessary
    lonmax = lon_convention(lon)
    if lon_convention(ps['lon'].values) != lonmax:
        ps = set_lon(ps, lonmax)

    # Interpolate ps onto lat-lon grid
    ps = interp_latlon(ps, lat, lon)

    # Add metadata to output DataArray
    ps.attrs = ds['ps'].attrs
    ps.attrs['title'] = ds.attrs['title']

    return ps

# ----------------------------------------------------------------------
def mask_below_topography():
    """Mask pressure level data below topography."""

# ----------------------------------------------------------------------
# Wrapper function to add topo field to a dataset
