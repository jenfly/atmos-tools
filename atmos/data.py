"""
Utility functions for atmospheric data wrangling / preparation.

- xray datasets and netCDF files
- Lat-lon geophysical data
- Topography
"""

import numpy as np
import collections
from mpl_toolkits import basemap
import xray
from xray import Dataset

from atmos.utils import print_if, print_odict, strictly_decreasing

# ======================================================================
# XRAY DATASETS AND NETCDF FILES
# ======================================================================

# ----------------------------------------------------------------------
def ds_print(ds, indent=2, width=20):
    """Print attributes of xray dataset and each of its variables."""
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
    """Display the attributes of data in a netcdf file."""
    with xray.open_dataset(filename, decode_cf=decode_cf) as ds:
        if verbose:
            ds_print(ds, indent, width)
        else:
            print(ds)


# ----------------------------------------------------------------------
def ds_unpack(dataset, missing_name=u'missing_value', offset_name=u'add_offset',
              scale_name=u'scale_factor', verbose=False, dtype=np.float64):
    """
    Unpack compressed data from an xray.Dataset object.

    Converts compressed int data to floats and missing values to NaN.
    Returns the results in an xray.Dataset object.
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
    missing values with NaN.  Returns data as an xray.Dataset object.
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
# LAT-LON GEOPHYSICAL DATA
# ======================================================================

# ----------------------------------------------------------------------
def latlon(data, latname='lat', lonname='lon'):
    """Return lat, lon ndarrays from DataArray."""
    return data[latname].values.copy(), data[lonname].values.copy()


# ----------------------------------------------------------------------
def latlon_equal(data1, data2, latname1='lat', lonname1='lon',
                 latname2='lat', lonname2='lon'):
    """Return True if input DataArrays have the same lat-lon coordinates."""

    lat1, lon1 = latlon(data1, latname=latname1, lonname=lonname1)
    lat2, lon2 = latlon(data2, latname=latname2, lonname=lonname2)
    is_equal = np.array_equal(lat1, lat2) and np.array_equal(lon1, lon2)
    return is_equal


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
def interp_latlon(data, lat_out, lon_out, lat_in=None, lon_in=None,
                  checkbounds=False, masked=False, order=1):
    """Interpolate data onto a new lat-lon grid.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Data to interpolate, with latitude as second-last dimension,
        longitude as last dimension.  Maximum array dimensions: 5-D.
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
    data_out : ndarray or xray.DataArray
        Data interpolated onto lat_out, lon_out grid
    """

    if isinstance(data, xray.DataArray):
        lat_in = data['lat'].values.copy()
        lon_in = data['lon'].values.copy()
        vals = data.values.copy()
    else:
        vals = data

    # Check for the common case that lat_in and/or lat_out are decreasing
    # and flip if necessary to work with basemap.interp()
    flip = False
    if strictly_decreasing(lat_in):
        lat_in = lat_in[::-1]
        vals = vals[...,::-1, :]
    if strictly_decreasing(lat_out):
        flip = True
        lat_out = lat_out[::-1]

    x_out, y_out = np.meshgrid(lon_out, lat_out)

    # Interp onto new lat-lon grid, iterating over all other dimensions
    # -- Remove the lat-lon dimensions (last 2 dimensions)
    dims = vals.shape
    dims = dims[:-2]
    ndim = len(dims)
    vals_out = np.empty(dims + x_out.shape)

    # Iterate over up to 3 additional dimensions
    if ndim > 3:
        raise ValueError('Too many dimensions in data.  Max 5-D.')
    if ndim == 3:
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    vals_out[i, j, k] = basemap.interp(
                        vals[i, j, k], lon_in, lat_in, x_out, y_out,
                        order=order, checkbounds=checkbounds, masked=masked)
    elif ndim == 2:
        for i in range(dims[0]):
            for j in range(dims[1]):
                vals_out[i, j] = basemap.interp(
                    vals[i, j], lon_in, lat_in, x_out, y_out, order=order,
                    checkbounds=checkbounds, masked=masked)
    elif ndim == 1:
        for i in range(dims[0]):
            vals_out[i] = basemap.interp(
                vals[i], lon_in, lat_in, x_out, y_out, order=order,
                checkbounds=checkbounds, masked=masked)
    else:
        vals_out = basemap.interp(
            vals, lon_in, lat_in, x_out, y_out, order=order,
            checkbounds=checkbounds, masked=masked)

    if flip:
        # Flip everything back to previous order
        vals_out = vals_out[...,::-1, :]
        lat_out = lat_out[::-1]

    if isinstance(data, xray.DataArray):
        coords = collections.OrderedDict()
        for key in data.coords.dims:
            if key != 'lat' and key != 'lon':
                coords[key] = data[key].values
        coords['lat'] = lat_out
        coords['lon'] = lon_out
        data_out = xray.DataArray(vals_out, name=data.name, coords=coords)
    else:
        data_out = vals_out

    return data_out


# ======================================================================
# TOPOGRAPHY
# ======================================================================

# ----------------------------------------------------------------------
def pres_units(units):
    """
    Return a standardized name (hPa or Pa) for the input pressure units.
    """
    hpa = ['mb', 'millibar', 'millibars', 'hpa', 'hectopascal', 'hectopascals']
    pa = ['pascal', 'pascals', 'pa']

    if units.lower() in hpa:
        return 'hPa'
    elif units.lower() in pa:
        return 'Pa'
    else:
        raise ValueError('Unknown units ' + units)


# ----------------------------------------------------------------------
def pres_pa(data, units):
    """Return pressure data (ndarray) in units of Pascals."""
    if pres_units(units) == 'hPa':
        data_out = data * 100
    elif pres_units(units) == 'Pa':
        data_out = data
    else:
        raise ValueError('Unknown units ' + units)
    return data_out


# ----------------------------------------------------------------------
def pres_hpa(data, units):
    """Return pressure data (ndarray) in units of hPa."""
    if pres_units(units) == 'hPa':
        data_out = data
    elif pres_units(units) == 'Pa':
        data_out = data / 100
    else:
        raise ValueError('Unknown units ' + units)
    return data_out


# ----------------------------------------------------------------------
def get_ps_clim(lat, lon, datafile='data/topo/ncep2_ps.nc'):
    """Return surface pressure climatology on selected lat-lon grid.

    Parameters
    ----------
    lat, lon : 1-D float array
        Latitude and longitude grid to interpolate surface pressure
        climatology onto.
    datafile : string, optional
        Name of file to read for surface pressure climatology.

    Returns
    -------
    ps : xray.DataArray
        DataArray of surface pressure climatology interpolated onto
        lat-lon grid.
    """

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
def correct_for_topography(data, topo_ps, plev=None, lat=None, lon=None):
    """Set pressure level data below topography to NaN.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Data to correct, with pressure, latitude, longitude as the
        last three dimensions.
    topo_ps : ndarray or xray.DataArray
        Climatological surface pressure to use for topography, on same
        lat-lon grid as data.
    plev, lat, lon : 1-D float array, optional
        Pressure levels, latitudes and longitudes of input data.
        Only used if data is an ndarray. If data is an xray.DataArray
        then plev, lat and lon are extracted from data.coords.

    Returns
    -------
    data_out : ndarray or xray.DataArray
        Data with grid points below topography set to NaN.
    """

    if isinstance(data, xray.DataArray):
        lat = data['lat'].values.copy()
        lon = data['lon'].values.copy()
        vals = data.values.copy()

        # Pressure levels in Pascals
        plev = data['plev'].values.copy()
        plev = pres_pa(plev, pres_units(data['plev'].units))
    else:
        vals = data

    if isinstance(topo_ps, xray.DataArray):
        if not latlon_equal(data, topo_ps):
            msg = 'Inputs data and topo_ps are not on same latlon grid.'
            raise ValueError(msg)

        # Surface pressure values in Pascals:
        ps_vals = topo_ps.values
        ps_vals = pres_pa(ps_vals, pres_units(topo_ps.units))
    else:
        ps_vals = topo_ps

    # For each vertical level, set any point below topography to NaN
    for k, p in enumerate(plev):
        ibelow = ps_vals < p
        vals[...,k,ibelow] = np.nan

    if isinstance(data, xray.DataArray):
        data_out = data.copy()
        data_out.values = vals
    else:
        data_out = vals

    return data_out


# ----------------------------------------------------------------------

# LAT-LON GEO
def average_over_box():
    """Return the data field averaged over a lat-lon box."""

def average_over_country():
    """Return the data field averaged over a country."""

def mask_ocean():
    """Return the data with ocean masked out."""

# PRESSURE / VERTICAL LEVELS

def near_surface():
    """Return the pressure-level data closest to surface."""

def interp_plevels():
    """Return the data interpolated onto new pressure level grid."""

def pressure_grid():
    """Return a grid of pressures from 1-D pressure level array."""

def pressure_grid_eta():
    """Return a grid of pressures for eta-level data."""

def ps_grid():
    """Return a grid of surface pressures."""

def int_pres():
    """Return the data integrated vertically by pressure."""
    # Maybe this should go in analysis.py

"""
TO DO:
- Edit interp_latlon so that it can handle data with additional dimensions.
"""
