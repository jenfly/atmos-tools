"""
Utility functions for atmospheric data wrangling / preparation.

- xray datasets and netCDF files
- Lat-lon wrangling
- Topography
"""

import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
from mpl_toolkits import basemap
import xray
from xray import Dataset

from atmos.utils import print_if, print_odict, strictly_decreasing

# ======================================================================
# XRAY DATASETS AND NETCDF FILES
# ======================================================================

# ----------------------------------------------------------------------
def ds_print(ds, indent=2, width=20):
    """
    Print metadata for xray dataset and for each variable.
    """
    line = '-' * 60

    # Attributes for dataset as a whole
    print('\n' + line + '\nDATASET\n' + line)
    print(ds)

    # Coordinates attributes
    print('\n' + line + '\nCOORDINATES\n' + line)
    for coord in ds.coords:
        print(coord)
        print_odict(ds[coord].attrs, indent=indent, width=width)

    # Data variables attributes
    print('\n' + line + '\nDATA VARIABLES\n' + line)
    for var in ds.data_vars:
        print(var)
        print_odict(ds[var].attrs, indent=indent, width=width)

    print(line + '\n')


# ----------------------------------------------------------------------
def ncdisp(filename, verbose=True, decode_cf=False, indent=2, width=20):
    """
    Display the attributes of data in a netcdf file.
    """
    with xray.open_dataset(filename, decode_cf=decode_cf) as ds:
        if verbose:
            ds_print(ds, indent, width)
        else:
            print(ds)


# ----------------------------------------------------------------------
def ds_unpack(dataset, missing_name=u'missing_value', offset_name=u'add_offset',
              scale_name=u'scale_factor', verbose=False, dtype=np.float64):
    """
    Unpack data from netcdf file as read with xray.open_dataset().

    Converts compressed int data to floats and missing values to NaN.
    """
    ds = dataset
    for var in ds.data_vars:
        print_if(var, verbose)
        vals = ds[var].values
        attrs = ds[var].attrs
        print_if(attrs, verbose, printfunc=print_odict)

        # Flag missing values for further processing
        if missing_name in attrs:
            missing_val = attrs[missing_name]
            imissing = vals == missing_val
            print_if('missing_val ' + str(missing_val), verbose)
            print_if('Found ' + str(imissing.sum()) + ' missings', verbose)
        else:
            imissing = []
            print_if('Missing values not flagged in input file', verbose)

        # Get offset and scaling factors, if any
        if offset_name in attrs:
            offset_val = attrs[offset_name]
            print_if('offset val ' + str(offset_val), verbose)
        else:
            offset_val = 0.0
            print_if('No offset in input file, setting to 0.0', verbose)
        if scale_name in attrs:
            scale_val = attrs[scale_name]
            print_if('scale val ' + str(scale_val), verbose)
        else:
            scale_val = 1.0
            print_if('No scaling in input file, setting to 1.0', verbose)

        # Convert from int to float with the offset and scaling
        vals = vals * scale_val + offset_val

        # Replace missing values with NaN
        vals[imissing] = np.nan

        # Replace the values in dataset with the converted ones
        ds[var].values = vals.astype(dtype)

    return ds


# ----------------------------------------------------------------------
def ncload(filename, verbose=True, unpack=True, missing_name=u'missing_value',
           offset_name=u'add_offset', scale_name=u'scale_factor',
           decode_cf=False):
    """
    Read data from netcdf file into xray dataset.

    If options are selected, unpacks from compressed form and/or replaces
    missing values with NaN.
    """
    with xray.open_dataset(filename, decode_cf=decode_cf) as ds:
        print_if('****** Reading file: ' + filename + '********', verbose)
        print_if(ds, verbose, printfunc=ds_print)
        if unpack:
            print_if('****** Unpacking data *********', verbose)
            ds = ds_unpack(ds, verbose=verbose, missing_name=missing_name,
                offset_name=offset_name, scale_name=scale_name)

        # Use the load() function so that the dataset is available after
        # the file is closed
        ds.load()
        return ds


# ======================================================================
# LAT-LON WRANGLING
# ======================================================================

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


# ======================================================================
# TOPOGRAPHY
# ======================================================================

# ----------------------------------------------------------------------
def get_topo(lat, lon, datafile='data/topo/ncep2_ps.nc'):
    """Return surface pressure climatology on selected latlon grid."""

    ds = ncload(datafile)
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
