"""
Utility functions for atmospheric data wrangling / preparation.

- ndarrays
- xray datasets and netCDF files
- Lat-lon geophysical data
- Topography
"""

import numpy as np
import collections
from mpl_toolkits import basemap
import xray
from xray import Dataset

from atmos.utils import print_if, print_odict, disptime
import atmos.utils as utils

# ======================================================================
# NDARRAYS
# ======================================================================

# ----------------------------------------------------------------------
def biggify(small, big, tile=False, debug=False):
    """Add singleton dimensions or tile an array for broadcasting.

    Parameters
    ----------
    small : ndarray
        Array which singleton dimensions will be added to.  Its
        dimensions must be a subset of big's dimensions.
    big : ndarray
        Array whose shape will be used to determine the shape of
        the output.
    tile : bool, optional
        If True, tile the array along the additional dimensions.
        If False, add singleton dimensions.
    debug : bool, optional
        Print debugging output.

    Returns
    -------
    biggified : ndarray
        Array of data from small, with dimensions added
        for any dimension that is in big but not in small.
    """

    dbig, dsmall = big.shape, small.shape

    # Check that all of the dimensions of small are contained within big
    check = [d in dbig or d == 1 for d in dsmall]
    if not np.all(check):
        msg = ('Dimensions of small ' + str(dsmall) +
            ' are not a subset of big ' + str(dbig))
        raise ValueError(msg)

    # Check that the dimensions appear in a compatible order
    inds = list()
    for d in dsmall:
        try:
            inds.append(dbig.index(d))
        except ValueError:
            inds.append(-1)
    if not utils.non_decreasing(inds):
        msg = ('Dimensions of small ' + str(dsmall) +
            ' are not in an order compatible with big ' + str(dbig))
        raise ValueError(msg)

    # Biggify the small array
    biggified = small
    ibig = big.ndim - 1
    ismall = small.ndim - 1
    n = -1

    # First add singleton dimensions
    while ismall >= 0 and ibig >= 0:
        print_if('ibig %d, ismall %d, n %d' % (ibig, ismall, n), debug)
        if dbig[ibig] == dsmall[ismall] or dsmall[ismall] == 1:
            print_if('  Same %d' % dbig[ibig], debug)
            ismall -= 1
        else:
            print_if('  Different.  Big %d, small %d' %
                (dbig[ibig], dsmall[ismall]), debug)
            biggified = np.expand_dims(biggified, n)
        n -= 1
        ibig -= 1

    # Expand with tiles if selected
    if tile:
        dims = list(biggified.shape)

        # First add any additional singleton dimensions needed to make
        # biggified of the same dimension as big\
        for i in range(len(dims), len(dbig)):
            dims.insert(0, 1)

        # Tile the array
        for i in range(-1, -1 - len(dims), -1):
            if dims[i] == dbig[i]:
                dims[i] = 1
            else:
                dims[i] = dbig[i]
        biggified = np.tile(biggified, dims)

    return biggified


# ----------------------------------------------------------------------
def nantrapz(y, x=None, axis=-1):
    """
    Integrate using the composite trapezoidal rule, ignoring NaNs

    Integrate `ym` (`x`) along given axis, where `ym` is a masked
    array of `y` with NaNs masked.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        If `x` is None, then spacing between all `y` elements is `dx`.
    axis : int, optional
        Specify the axis.

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule.
    """

    ym = np.ma.masked_array(y, np.isnan(y))
    trapz = np.trapz(ym, x, axis=axis)

    # Convert from masked array back to regular ndarray
    if isinstance(trapz, np.ma.masked_array):
        trapz = trapz.filled(np.nan)

    return trapz


# ======================================================================
# XRAY DATASETS AND FILE I/O
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


# ----------------------------------------------------------------------
def load_concat(paths, var, concat_dim=None, verbose=False):
    """Load a variable from multiple files and concatenate into one.

    Especially useful for extracting variables split among multiple
    OpenDAP files.

    *** Bug:  doesn't work when concat_dim=None.  Fix later when
    this functionality is needed. ***

    Parameters
    ----------
    paths : list of strings
        List of file paths or OpenDAP urls to process.
    var : str
        Name of variable to extract.
    concat_dim : str, optional
        Dimension to concatenate along.  If None, a new dimension is
        created for concatenation.
    verbose : bool, optional
        If True, print updates while processing files.

    Returns:
    --------
    data : xray.DataArray
        Data extracted from input files.
    """

    pieces = list()
    for p in paths:
        print_if(None, verbose, printfunc=disptime)
        print_if('Loading ' + p, verbose)
        with xray.open_dataset(p) as ds:
            print_if('Appending data', verbose)
            pieces.append(ds[var].load())

    print_if('Concatenating data', verbose)
    data = xray.concat(pieces, dim=concat_dim)
    print_if(None, verbose, printfunc=disptime)
    return data


# ----------------------------------------------------------------------
def _subset_1dim(data, dim_name, lower_or_list, upper=None,
                 incl_lower=True, incl_upper=True):
    """Extract a subset of a DataArray along a named dimension."""

    vals = data[dim_name]
    if upper is None:
        valrange = lower_or_list
    else:
        if incl_lower:
            ind1 = vals >= lower_or_list
        else:
            ind1 = vals > lower_or_list
        if incl_upper:
            ind2 = vals <= upper
        else:
            ind2 = vals < upper
        valrange = vals[ind1 & ind2]

    return data.sel(**{dim_name : valrange}).copy()


def subset(data, dim_name, lower_or_list, upper=None,
           dim_name2=None, lower_or_list2=None, upper2=None,
           incl_lower=True, incl_upper=True):
    """Extract a subset of a DataArray along 1 or 2 named dimensions.

    Returns a DataArray sub extracted from input data, such that:
        sub[dim_name] >= lower_or_list & sub[dim_name] <= upper,
    OR  sub[dim_name] == lower_or_list (if lower_or_list is a list)
    And similarly for dim_name2, if included.

    Parameters
    ----------
    data : xray.DataArray
        Data source for extraction.
    dim_name : string
        Name of dimension to extract from.
    lower_or_list : scalar or list of int or float
        If scalar, then used as the lower bound for the   subset range.
        If list, then the subset matching the list will be extracted.
    upper : int or float, optional
        Upper bound for subset range.
    dim_name2, lower_or_list2, upper2 : optional
        Parameters as described above for optional 2nd dimension.
    incl_lower, incl_upper : bool, optional
        If True lower / upper bound is inclusive, with >= or <=.
        If False, lower / upper bound is exclusive with > or <.
        If lower_or_list is a list, then the whole list is included
        and these parameters are ignored.

    Returns
    -------
        sub : xray.DataArray
    """

    sub = _subset_1dim(data, dim_name, lower_or_list, upper, incl_lower,
                       incl_upper)

    if dim_name2 is not None:
        sub = _subset_1dim(sub, dim_name2, lower_or_list2, upper2, incl_lower,
                           incl_upper)

    return sub


# ----------------------------------------------------------------------
def coords_init(data):
    """Return OrderedDict of xray.DataArray-like coords for numpy array.

    Parameters
    ----------
    data : ndarray

    Returns
    -------
    coords : collections.OrderedDict
        Keys are dim_0, dim_1, etc. and values are np.arange(d) for
        each value d in data.shape.
    """

    coords = collections.OrderedDict()
    dims = data.shape

    for i, d in enumerate(dims):
        coords['dim_' + str(i)] = np.arange(d)

    return coords


# ----------------------------------------------------------------------
def coords_assign(coords, dim, new_name, new_val):
    """Reassign an xray.DataArray-style coord at a given dimension.

    Parameters
    ----------
    coords : collections.OrderedDict
        Ordered dictionary of coord name : value pairs.
    dim : int
        Dimension to change (e.g. -1 for last dimension).
    new_name : string
        New name for coordinate key.
    new_val : any
        New value, e.g. numpy array of values

    Returns
    -------
    new_coords : collections.OrderedDict
        Ordered dictionary with altered dimension.

    Example
    -------
    lat = np.arange(89., -89., -1.0)
    lon = np.arange(0., 359., 1.)
    data = np.ones((len(lat), len(lon)), dtype=float)
    coords = coords_init(data)
    coords = coords_assign(coords, -1, 'lon', lon)
    coords = coords_assign(coords, -2, 'lat', lat)
    """

    items = list(coords.items())
    items[dim] = (new_name, new_val)
    new_coords = collections.OrderedDict(items)
    return new_coords


# ======================================================================
# LAT-LON GEOPHYSICAL DATA
# ======================================================================

# ----------------------------------------------------------------------
def get_lat(data, latname=None, return_name=False):
    """Return latitude array (or dimension name) from DataArray.

    Parameters
    ----------
    data : xray.DataArray
        Data array to search for latitude coords.
    latname : string, optional
        Name of latitude coord in data.  If omitted, search through
        a list of common names for a match.
    return_name : bool, optional
        Return the name of the latitude dimension rather than the
        array of values.

    Returns
    -------
    lat : ndarray or string
        Latitude array or name of latitude dimension

    Notes
    -----
    The latitude dimension names searched for are:
      latnames = ['lat', 'lats', 'latitude', 'YDim','Y', 'y']
    """

    latnames = ['lat', 'lats', 'latitude', 'YDim','Y', 'y']

    if latname is None:
        # Look for lat names in data coordinates
        found = [i for i, s in enumerate(latnames) if s in data.coords]

        if len(found) == 0:
            raise ValueError("Can't find latitude names in data coords %s" %
                             data.coords.keys())
        if len(found) > 1:
            raise ValueError('Conflicting possible latitude names in coords %s'
                % data.coords.keys())
        else:
            latname = latnames[found[0]]

    lat = data[latname].values.copy()

    if return_name:
        return latname
    else:
        return lat


# ----------------------------------------------------------------------
def get_lon(data, lonname=None, return_name=False):
    """Return longitude array (or dimension name) from DataArray.

    Parameters
    ----------
    data : xray.DataArray
        Data array to search for longitude coords.
    lonname : string, optional
        Name of longitude coords in data.  If omitted, search through
        a list of common names for a match.
    return_name : bool, optional
        Return the name of the longitude dimension rather than the
        array of values.

    Returns
    -------
    lon : ndarray or string
        Longitude array or dimension name

    Notes
    -----
    The longitude dimension names searched for are:
      lonnames = ['lon', 'long', 'lons', 'longitude', 'XDim', 'X', 'x']
    """

    lonnames = ['lon', 'long', 'lons', 'longitude', 'XDim', 'X', 'x']

    if lonname is None:
        # Look for longitude names in data coordinates
        found = [i for i, s in enumerate(lonnames) if s in data.coords]

        if len(found) == 0:
            raise ValueError("Can't find longitude names in data coords %s" %
                             data.coords.keys())
        if len(found) > 1:
            raise ValueError('Conflicting possible longitude names in coords %s'
                % data.coords.keys())
        else:
            lonname = lonnames[found[0]]

    lon = data[lonname].values.copy()

    if return_name:
        return lonname
    else:
        return lon


# ----------------------------------------------------------------------
def latlon_equal(data1, data2, latname1=None, lonname1=None,
                 latname2=None, lonname2=None):
    """Return True if input DataArrays have the same lat-lon coordinates."""

    lat1, lon1 = get_lat(data1, latname1), get_lon(data1, lonname1)
    lat2, lon2 = get_lat(data2, latname2), get_lon(data2, lonname2)
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
def set_lon(data, lonmax=360, lon=None, lonname='lon'):
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
    lonname : string, optional
        Name of longitude coordinate in data, if data is a DataArray

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
        lon = data[lonname]
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
        data_out[lonname].values = lon_out
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
    lat_out, lon_out : 1-D float or int array
        Latitude and longitudes to interpolate onto.
    lat_in, lon_in : 1-D float or int array, optional
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
        lat_in, lon_in = get_lat(data), get_lon(data)
        vals = data.values.copy()
    else:
        vals = data

    # Check for the common case that lat_in and/or lat_out are decreasing
    # and flip if necessary to work with basemap.interp()
    flip = False
    if utils.strictly_decreasing(lat_in):
        lat_in = lat_in[::-1]
        vals = vals[...,::-1, :]
    if utils.strictly_decreasing(lat_out):
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


# ----------------------------------------------------------------------
def mask_oceans(data, lat=None, lon=None, inlands=True, resolution='l',
                grid=5):
    """Return the data with ocean grid points set to NaN.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Data to mask, with latitude as second-last dimension,
        longitude as last dimension.  Maximum array dimensions: 5-D.
    lat, lon : ndarray, optional
        Latitude and longitude arrays.  Only used if data is an
        ndarray and not an xray.DataArray.
    inlands : bool, optional
        If False, mask only ocean points and not inland lakes.
    resolution : {'c','l','i','h', 'f'}, optional
        gshhs coastline resolution used to define land/sea mask.
    grid : {1.25, 2.5, 5, 10}, optional
        Land/sea mask grid spacing in minutes.

    Returns
    -------
    data_out : ndarray or xray.DataArray
        Data with ocean grid points set to NaN.
    """

    if isinstance(data, xray.DataArray):
        lat, lon = get_lat(data), get_lon(data)
        vals = data.values.copy()
    else:
        vals = data

    x, y = np.meshgrid(lon, lat)
    dims = vals.shape
    ndim = vals.ndim
    vals_out = np.ones(vals.shape, dtype=float)
    vals_out = np.ma.masked_array(vals_out, np.isnan(vals_out))

    # Iterate over up to 3 additional dimensions
    if ndim > 5:
        raise ValueError('Too many dimensions in data.  Max 5-D.')
    if ndim == 5:
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    vals_out[i, j, k] = basemap.maskoceans(x, y, vals[i, j, k],
                        inlands=inlands, resolution=resolution, grid=grid)
    elif ndim == 4:
        for i in range(dims[0]):
            for j in range(dims[1]):
                vals_out[i, j] = basemap.maskoceans(x, y, vals[i, j],
                    inlands=inlands, resolution=resolution, grid=grid)
    elif ndim == 3:
        for i in range(dims[0]):
            vals_out[i] = basemap.maskoceans(x, y, vals[i],
                inlands=inlands, resolution=resolution, grid=grid)

    else:
        vals_out = basemap.maskoceans(x, y, vals,
            inlands=inlands, resolution=resolution, grid=grid)

    # Convert from masked array to regular array with NaNs
    vals_out = vals_out.filled(np.nan)

    if isinstance(data, xray.DataArray):
        data_out = xray.DataArray(vals_out, name=data.name, coords=data.coords)
    else:
        data_out = vals_out

    return data_out


# ----------------------------------------------------------------------
def mean_over_geobox(data, lat1, lat2, lon1, lon2, lat=None, lon=None,
                     area_wtd=True, land_only=False):
    """Return the mean of an array over a lat-lon region.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Data to average, with latitude as second-last dimension and
        longitude as last dimension.
    lat1, lat2, lon1, lon2 : float
        Latitude and longitude limits for averaging region, with
        lon1 < lon2 and lat1 < lat2.
    lat, lon : ndarray, optional
        Latitude and longitude arrays.  Only used if data is an
        ndarray and not an xray.DataArray.
    area_wtd : bool, optional
        Return the area-weighted average (weighted by cos(lat))
    land_only : bool, optional
        Mask out ocean grid points so that only data over land is
        included in the mean.

    Returns
    -------
    avg : ndarray or xray.DataArray
        The data averaged over the lat-lon region.
    """

    if not isinstance(data, xray.DataArray):
        if lat is None or lon is None:
            raise ValueError('Latitude and longitude arrays must be provided '
                'if data is not an xray.DataArray.')
        latname, lonname = 'lat', 'lon'
        coords = coords_init(data)
        coords = coords_assign(coords, -1, lonname, lon)
        coords = coords_assign(coords, -2, latname, lat)
        data_out = xray.DataArray(data, coords=coords)
    else:
        data_out = data
        attrs = data.attrs
        coords = data.coords
        dims = data.dims[:-2]
        latname = get_lat(data, return_name=True)
        lonname = get_lon(data, return_name=True)

    data_out = subset(data_out, latname, lat1, lat2, lonname, lon1, lon2)

    if land_only:
        data_out = mask_oceans(data_out)

    # Mean over longitudes
    data_out = data_out.mean(axis=-1)

    # Array of latitudes with same NaN mask as the data so that the
    # area calculation is correct
    lat_rad = np.radians(get_lat(data_out))
    lat_rad = biggify(lat_rad, data_out, tile=True)
    mdat = np.ma.masked_array(data_out, np.isnan(data_out))
    lat_rad = np.ma.masked_array(lat_rad, mdat.mask)
    lat_rad = lat_rad.filled(np.nan)

    if area_wtd:
        # Weight by area with cos(lat)
        coslat = np.cos(lat_rad)
        data_out = data_out * coslat
        area = nantrapz(coslat, lat_rad, axis=-1)
    else:
        area = nantrapz(np.ones(lat_rad.shape, dtype=float), lat_rad, axis=-1)

    # Mean over latitudes
    avg = nantrapz(data_out, lat_rad, axis=-1) / area

    # Pack output into DataArray with the metadata that was lost in np.trapz
    if isinstance(data, xray.DataArray) and not isinstance(avg, xray.DataArray):
        avg = xray.DataArray(avg, dims=dims, attrs=attrs)
        for d in dims:
            avg[d] = coords[d]

    return avg


# ======================================================================
# PRESSURE LEVEL DATA AND TOPOGRAPHY
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
def pres_convert(pres, units_in, units_out):
    """Convert pressure array from units_in and return in units_out."""

    if pres_units(units_in) == pres_units(units_out):
        pres_out = pres
    elif pres_units(units_in) == 'hPa' and pres_units(units_out) == 'Pa':
        pres_out = pres * 100
    elif pres_units(units_in) == 'Pa' and pres_units(units_out) == 'hPa':
        pres_out = pres / 100
    else:
        raise ValueError('Problem with input/output units.')
    return pres_out


# ----------------------------------------------------------------------
def get_plev(data, plevname=None, units='hPa', return_name=False):
    """Return pressure level array (or dimension name) from DataArray.

    Parameters
    ----------
    data : xray.DataArray
        Data array to search for pressure level coords.
    lonname : string, optional
        Name of pressure level coords in data.  If omitted, search
        through a list of common names for a match.
    units: string, optional
        Pressure units to use for output.
    return_name : bool, optional
        Return the name of the pressure dimension rather than the
        array of values.

    Returns
    -------
    plev : ndarray or string
        Pressure level array or dimension name

    Notes
    -----
    The pressure level dimension names searched for are:
      plevnames = ['plev', 'plevel', 'plevels', 'Height']
    """

    plevnames = ['plev', 'plevel', 'plevels', 'Height']

    if plevname is None:
        # Look for pressure level names in data coordinates
        found = [i for i, s in enumerate(plevnames) if s in data.coords]

        if len(found) == 0:
            raise ValueError(
                "Can't find presure level names in data coords %s" %
                             data.coords.keys())
        if len(found) > 1:
            raise ValueError(
                'Conflicting possible pressure level names in coords %s'
                % data.coords.keys())
        else:
            plevname = plevnames[found[0]]

    plev = data[plevname].values.copy()
    plev = pres_convert(plev, data[plevname].units, units)

    if return_name:
        return plevname
    else:
        return plev


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
        plev = pres_convert(plev, data['plev'].units, 'Pa')
    else:
        vals = data

    if isinstance(topo_ps, xray.DataArray):
        if not latlon_equal(data, topo_ps):
            msg = 'Inputs data and topo_ps are not on same latlon grid.'
            raise ValueError(msg)

        # Surface pressure values in Pascals:
        ps_vals = topo_ps.values
        ps_vals = pres_convert(ps_vals, topo_ps.units, 'Pa')
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
def near_surface(data, pdim=-3, return_inds=False):
    """Return the pressure-level data closest to surface.

    At each grid point, the first non-NaN level is taken as the
    near-surface level.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Input data, maximum of 5 dimensions.  Pressure levels must
        be the last, second-last or third-last dimension.
    pdim : {-3, -2, -1}, optional
        Dimension of vertical levels in data.
    return_inds : bool, optional
        If True, return the pressure-level indices of the extracted
        data in a tuple along with the near-surface data.
        If False, return only the near-surface data.

    Returns
    -------
    data_s[, ind_s] : ndarray or xray.DataArray[, ndarray]
        Near-surface data [and indices of extracted data, if
        return_inds is True]. If input data is an xray.DataArray,
        data_s is returned as an xray.DataArray, otherwise as
        an ndarray.
    """

    # Maximum number of dimensions handled by this code
    nmax = 5
    ndim = data.ndim

    if ndim > 5:
        raise ValueError('Input data has too many dimensions. Max 5-D.')

    # Save metadata for output DataArray, if applicable
    if isinstance(data, xray.DataArray):
        i_DataArray = True
        data = data.copy()
        title = 'Near-surface data extracted from pressure level data'
        attrs = collections.OrderedDict({'title' : title})
        for d in data.attrs:
            attrs[d] = data.attrs[d]
        pname = get_plev(data, return_name=True)
        coords = collections.OrderedDict()
        for key in data.dims:
            if key != pname:
                coords[key] = data.coords[key]
    else:
        i_DataArray = False

    # Add singleton dimensions for looping, if necessary
    for i in range(ndim, nmax):
        data = np.expand_dims(data, axis=0)

    # Make sure pdim is indexing from end
    pdim_in = pdim
    if pdim > 0:
        pdim = pdim - nmax

    # Iterate over all other dimensions
    dims = list(data.shape)
    dims.pop(pdim)
    data_s = np.nan*np.ones(dims, dtype=float)
    ind_s = np.ones(dims, dtype=int)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                for m in range(dims[3]):
                    if pdim == -3:
                        sub = data[i,j,:,k,m]
                    elif pdim == -2:
                        sub = data[i,j,k,:,m]
                    elif pdim == -1:
                        sub = data[i,j,k,m,:]
                    else:
                        raise ValueError('Invalid p dimension ' + str(pdim_in))
                    ind = np.where(~np.isnan(sub))[0][0]
                    data_s[i,j,k,m] = sub[ind]
                    ind_s[i,j,k,m] = ind

    # Collapse any additional dimensions that were added
    for i in range(ndim - 1, data_s.ndim):
        data_s = data_s[0]
        ind_s = ind_s[0]

    # Pack data_s into an xray.DataArray if input was in that form
    if i_DataArray:
        data_s = xray.DataArray(data_s, coords=coords, attrs=attrs)

    # Return data only, or tuple of data plus array of indices extracted
    if return_inds:
        return data_s, ind_s
    else:
        return data_s

# ----------------------------------------------------------------------

# LAT-LON GEO
# def average_over_country():
#    """Return the data field averaged over a country."""



def interp_plevels():
    """Return the data interpolated onto new pressure level grid."""


def int_pres():
    """Return the data integrated vertically by pressure."""
