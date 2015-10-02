"""
Utility functions for atmospheric data wrangling / preparation.

- ndarrays
- netCDF files
- Lat-lon geophysical data
- Pressure level data and topography
"""

from __future__ import division
import numpy as np
import collections
import scipy.interpolate as interp
from mpl_toolkits import basemap
import xray
from xray import Dataset
import time

from atmos.utils import print_if, disptime
import atmos.utils as utils
import atmos.xrhelper as xr
#from atmos.xrhelper import subset
from atmos.constants import const as constants

# ======================================================================
# NDARRAYS
# ======================================================================

# ----------------------------------------------------------------------
def biggify(small, big, tile=False):
    """Add dimensions or tile an array for broadcasting.

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

    Returns
    -------
    biggified : ndarray
        Array of data from small, with dimensions added
        for any dimension that is in big but not in small.
    """

    debug = False
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
# COORDINATES AND SUBSETS
# ======================================================================

# ----------------------------------------------------------------------
def get_coord(data, coord_type=None, return_type='values', coord_name=None):
    """Return values, name or dimension of coordinate in DataArray.

    Parameters
    ----------
    data : xray.DataArray
        Data array to search for latitude coords.
    coord_type : {'lat', 'lon', 'plev'}, optional
        Type of coordinate to extract.  If omitted, then coord_name
        must be provided.
    return_type : {'values', 'name', 'dim'}, optional
        'values' : Return an array of coordinate values.
        'name' : Return the name of the coordinate.
        'dim' : Return the dimension of the coordinate.
    coord_name : string, optional
        Name of coordinate in data.  If omitted, search through
        a list of common names for a match.

    Returns
    -------
    output : ndarray, string or int

    The coordinate names searched through are:
    'lat' : ['lat', 'lats', 'latitude', 'YDim','Y', 'y']
    'lon' : ['lon', 'long', 'lons', 'longitude', 'XDim', 'X', 'x']
    'plev' : ['plev', 'plevel', 'plevels', 'Height']
    """

    names_all = dict()
    names_all['lat'] = ['lat', 'lats', 'latitude', 'YDim','Y', 'y']
    names_all['lon'] = ['lon', 'long', 'lons', 'longitude', 'XDim', 'X', 'x']
    names_all['plev'] = ['plev', 'plevel', 'plevels', 'Height']

    # Look in list of common coordinate names
    if coord_name is None:
        if coord_type.lower() not in names_all.keys():
            raise ValueError('Invalid coord_type ' + coord_type + '. '
                'Valid coord_types are ' + ', '.join(names_all.keys()))
        names = names_all[coord_type.lower()]
        found = [i for i, s in enumerate(names) if s in data.coords]

        if len(found) == 0:
            raise ValueError("Can't find coordinate name in data coords %s" %
                             data.coords.keys())
        if len(found) > 1:
            raise ValueError('Conflicting possible coord names in coords %s'
                % data.coords.keys())
        else:
            coord_name = names[found[0]]

    if return_type == 'values':
        output = data[coord_name].values.copy()
    elif return_type == 'name':
        output = coord_name
    elif return_type == 'dim':
        output = data.dims.index(coord_name)
    else:
        raise ValueError('Invalid return_type ' + return_type)

    return output


# ----------------------------------------------------------------------
def subset(data, dim_name, lower_or_list, upper=None,
           dim_name2=None, lower_or_list2=None, upper2=None,
           incl_lower=True, incl_upper=True, search=True):
    """Extract a subset of a DataArray or Dataset along named dimensions.

    Returns a DataArray or Dataset sub extracted from input data,
    such that:
        sub[dim_name] >= lower_or_list & sub[dim_name] <= upper,
    OR  sub[dim_name] == lower_or_list (if lower_or_list is a list)
    And similarly for dim_name2, if included.

    This function calls atmos.xrhelper.subset with the additional
    feature of calling the get_coord function to find common
    dimension names (e.g. 'XDim' for latitude)

    Parameters
    ----------
    data : xray.DataArray or xray.Dataset
        Data source for extraction.
    dim_name : string
        Name of dimension to extract from.  If dim_name is not in
        data.dims, then the get_coord() function is used
        to search for a similar dimension name (if search is True).
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
    search : bool, optional
        If True, call the get_coord function if dim_name is not found
        in the dimension names of data.

    Returns
    -------
        sub : xray.DataArray or xray.Dataset
    """

    if search:
        nms = ['lat', 'lon', 'plev']
        if dim_name in nms and dim_name not in data.dims:
            dim_name = get_coord(data, dim_name, 'name')
        if dim_name2 in nms and dim_name2 not in data.dims:
            dim_name2 = get_coord(data, dim_name2, 'name')

    return xr.subset(data, dim_name, lower_or_list, upper, dim_name2,
                    lower_or_list2, upper2, incl_lower, incl_upper)


# ======================================================================
# NETCDF FILE I/O
# ======================================================================

# ----------------------------------------------------------------------
def ncdisp(filename, verbose=True, decode_cf=False, indent=2, width=None):
    """Display the attributes of data in a netcdf file."""
    with xray.open_dataset(filename, decode_cf=decode_cf) as ds:
        if verbose:
            xr.ds_print(ds, indent, width)
        else:
            print(ds)


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
        print_if(ds, verbose, printfunc=xr.ds_print)
        if unpack:
            print_if('****** Unpacking data *********', verbose)
            ds = xr.ds_unpack(ds, verbose=verbose, missing_name=missing_name,
                offset_name=offset_name, scale_name=scale_name)

        # Use the load() function so that the dataset is available after
        # the file is closed
        ds.load()
        return ds


# ----------------------------------------------------------------------
def load_concat(paths, var, concat_dim='TIME', subset1=(None, None, None),
                subset2=(None, None, None), verbose=True):
    """Load a variable from multiple files and concatenate into one.

    Especially useful for extracting variables split among multiple
    OpenDAP files.

    *** Note:  doesn't work when concat_dim=None.  Add this in future
    if needed, so that concat_dim=None creates a new dimension for
    concatenation. ***

    Parameters
    ----------
    paths : list of strings
        List of file paths or OpenDAP urls to process.
    var : str
        Name of variable to extract.
    concat_dim : str
        Name of dimension to concatenate along.
    subset1, subset2 : (str, float(s), float(s)), optional
        Tuple to indicate subset(s) to extract, in the form:
        (dim_name, lower_or_list, upper)
        e.g. subset1 = ('lon', 0, 120)
             subset2 = ('lat', -45, 45)
        e.g. subset1 = ('plev', 200, 200)
        The dimension name can be the actual dimension name
        (e.g. 'XDim') or a generic name (e.g. 'lon') and get_coord()
        is called to find the specific name.
    verbose : bool, optional
        If True, print updates while processing files.

    Returns:
    --------
    data : xray.DataArray
        Data extracted from input files.
    """

    # Number of times to attempt opening file (in case of server problems)
    NMAX = 3
    # Wait time (seconds) between attempts
    WAIT = 60

    pieces = list()
    for p in paths:
        print_if(None, verbose, printfunc=disptime)
        print_if('Loading ' + p, verbose)
        attempt = 0
        while attempt < NMAX:
            try:
                with xray.open_dataset(p) as ds:
                    print_if('Appending data', verbose)
                    piece = ds[var]
                    if subset1[0] is not None:
                        piece = subset(piece, subset1[0], subset1[1],
                                       subset1[2], subset2[0], subset2[1],
                                       subset2[2])
                    pieces.append(piece.load())
                    attempt = NMAX
            except RuntimeError as err:
                attempt += 1
                if attempt < NMAX:
                    print('File error.  Attempting again in %d s' % WAIT)
                    time.sleep(WAIT)
                else:
                    raise err

    print_if('Concatenating data', verbose)
    data = xray.concat(pieces, dim=concat_dim)
    print_if(None, verbose, printfunc=disptime)
    return data


# ----------------------------------------------------------------------
def save_nc(filename, *args):
    """Save xray.DataArray variables to a netcdf file.

    Call Signatures
    ---------------
    save_nc(filename, var1)
    save_nc(filename, var1, var2)
    save_nc(filename, var1, var2, var3)
    etc...

    Parameters
    ----------
    filename : string
        File path for saving.
    var1, var2, ... : xray.DataArrays
        List of xray.DataArrays with compatible coordinates.
    """

    ds = xr.vars_to_dataset(*args)
    ds.to_netcdf(filename)
    return None



# ======================================================================
# LAT-LON GEOPHYSICAL DATA
# ======================================================================

# ----------------------------------------------------------------------
def latlon_equal(data1, data2, latname1=None, lonname1=None,
                 latname2=None, lonname2=None):
    """Return True if input DataArrays have the same lat-lon coordinates."""

    lat1 = get_coord(data1, 'lat', coord_name=latname1)
    lon1 = get_coord(data1, 'lon', coord_name=lonname1)
    lat2 = get_coord(data2, 'lat', coord_name=latname2)
    lon2 = get_coord(data2, 'lon', coord_name=lonname2)

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
def set_lon(data, lonmax=360, lon=None, lonname=None):
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
        lon = get_coord(data, 'lon', coord_name=lonname)
        if lonname is None:
            lonname = get_coord(data, 'lon', 'name')
        coords, attrs, name = xr.meta(data)
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
        coords[lonname].values = lon_out
        data_out = xray.DataArray(vals_out, name=name, coords=coords,
                                  attrs=attrs)
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

    # Maximum number of dimensions handled by this code
    nmax = 5
    ndim = data.ndim

    if ndim > 5:
        raise ValueError('Input data has too many dimensions. Max 5-D.')

    if isinstance(data, xray.DataArray):
        lat_in = get_coord(data, 'lat')
        latname = get_coord(data, 'lat', 'name')
        lon_in = get_coord(data, 'lon')
        lonname = get_coord(data, 'lon', 'name')
        coords, attrs, name = xr.meta(data)
        coords[latname] = xray.DataArray(lat_out, coords={latname : lat_out},
                                         attrs=data[latname].attrs)
        coords[lonname] = xray.DataArray(lon_out, coords={lonname : lon_out},
                                         attrs=data[lonname].attrs)
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

    # Initialize output array
    dims = vals.shape
    dims = dims[:-2]
    vals_out = np.empty(dims + x_out.shape)

    # Add singleton dimensions for looping, if necessary
    for i in range(ndim, nmax):
        vals = np.expand_dims(vals, axis=0)
        vals_out = np.expand_dims(vals_out, axis=0)

    # Interp onto new lat-lon grid, iterating over all other dimensions
    dims = vals_out.shape[:-2]
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                vals_out[i, j, k] = basemap.interp(
                    vals[i, j, k], lon_in, lat_in, x_out, y_out,
                    order=order, checkbounds=checkbounds, masked=masked)

    # Collapse any additional dimensions that were added
    for i in range(ndim, vals_out.ndim):
        vals_out = vals_out[0]

    if flip:
        # Flip everything back to previous order
        vals_out = vals_out[...,::-1, :]
        lat_out = lat_out[::-1]

    if isinstance(data, xray.DataArray):
        data_out = xray.DataArray(vals_out, name=name, coords=coords,
                                  attrs=attrs)
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

    # Maximum number of dimensions handled by this code
    nmax = 5
    ndim = data.ndim

    if ndim > 5:
        raise ValueError('Input data has too many dimensions. Max 5-D.')

    if isinstance(data, xray.DataArray):
        lat = get_coord(data, 'lat')
        lon = get_coord(data, 'lon')
        coords, attrs, name = xr.meta(data)
        vals = data.values.copy()
    else:
        vals = data

    # Convert to 180W-180E convention that basemap.maskoceans requires
    lonmax = lon_convention(lon)
    if lonmax == 360:
        vals, lon = set_lon(vals, lonmax=180, lon=lon)

    # Add singleton dimensions for looping, if necessary
    for i in range(ndim, nmax):
        vals = np.expand_dims(vals, axis=0)

    # Initialize output
    vals_out = np.ones(vals.shape, dtype=float)
    vals_out = np.ma.masked_array(vals_out, np.isnan(vals_out))

    # Mask oceans, iterating over additional dimensions
    x, y = np.meshgrid(lon, lat)
    dims = vals_out.shape[:-2]
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                vals_out[i, j, k] = basemap.maskoceans(
                    x, y, vals[i, j, k], inlands=inlands,
                    resolution=resolution, grid=grid)

    # Convert from masked array to regular array with NaNs
    vals_out = vals_out.filled(np.nan)

    # Collapse any additional dimensions that were added
    for i in range(ndim, vals_out.ndim):
        vals_out = vals_out[0]

    # Convert back to original longitude convention
    if lonmax == 360:
        vals_out, lon = set_lon(vals_out, lonmax=lonmax, lon=lon)

    if isinstance(data, xray.DataArray):
        data_out = xray.DataArray(vals_out, name=name, coords=coords,
                                  attrs=attrs)
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
        coords = xr.coords_init(data)
        coords = xr.coords_assign(coords, -1, lonname, lon)
        coords = xr.coords_assign(coords, -2, latname, lat)
        data_out = xray.DataArray(data, coords=coords)
    else:
        data_out = data
        coords, attrs, name = xr.meta(data)
        latname = get_coord(data, 'lat', 'name')
        lonname = get_coord(data, 'lon', 'name')
        coords = utils.odict_delete(coords, latname)
        coords = utils.odict_delete(coords, lonname)

    if land_only:
        data_out = mask_oceans(data_out)

    data_out = subset(data_out, latname, lat1, lat2, lonname, lon1, lon2)

    # Mean over longitudes
    data_out = data_out.mean(axis=-1)

    # Array of latitudes with same NaN mask as the data so that the
    # area calculation is correct
    lat_rad = np.radians(get_coord(data_out, 'lat'))
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
        avg = xray.DataArray(avg, name=name, coords=coords, attrs=attrs)

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
    ps.attrs = utils.odict_insert(ps.attrs, 'title', ds.attrs['title'], pos=0)

    # Check what longitude convention is used in the surface pressure
    # climatology and switch if necessary
    lonmax = lon_convention(lon)
    lon_ps = get_coord(ps, 'lon')
    if lon_convention(lon_ps) != lonmax:
        ps = set_lon(ps, lonmax)

    # Interpolate ps onto lat-lon grid
    ps = interp_latlon(ps, lat, lon)

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
        lat = get_coord(data, 'lat')
        lon = get_coord(data, 'lon')
        coords, attrs, name = xr.meta(data)
        vals = data.values.copy()
        # -- Pressure levels in Pascals
        plev = get_coord(data, 'plev')
        pname = get_coord(data, 'plev', 'name')
        plev = pres_convert(plev, data[pname].units, 'Pa')
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
        data_out = xray.DataArray(vals, name=name, coords=coords, attrs=attrs)
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

    if ndim > nmax:
        raise ValueError('Input data has too many dimensions. Max 5-D.')

    # Save metadata for output DataArray, if applicable
    if isinstance(data, xray.DataArray):
        i_DataArray = True
        data = data.copy()
        coords, attrs, name = xr.meta(data)
        title = 'Near-surface data extracted from pressure level data'
        attrs = utils.odict_insert(attrs, 'title', title, pos=0)
        pname = get_coord(data, 'plev', 'name')
        del(coords[pname])
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
        data_s = xray.DataArray(data_s, name=name, coords=coords, attrs=attrs)

    # Return data only, or tuple of data plus array of indices extracted
    if return_inds:
        return data_s, ind_s
    else:
        return data_s


# ----------------------------------------------------------------------
def interp_plevels(data, plev_new, plev_in=None, pdim=-3, kind='linear'):
    """Return the data interpolated onto new pressure level grid.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Input data, maximum of 5 dimensions.  Pressure levels must
        be the last, second-last or third-last dimension.
    plev_new : ndarray
        New pressure levels to interpolate onto.
    plev_in : ndarray
        Original pressure levels of data.  If data is an xray.DataArray,
        then the values from data.coords are used.
    pdim : {-3, -2, -1}, optional
        Dimension of vertical levels in data.
    kind : string, optional
        Type of interpolation, e.g. 'linear', 'cubic', 'nearest', etc.
        See scipy.interpolate.interp1d for all options.

    Returns
    -------
    data_i : ndarray or xray.DataArray
        Interpolated data. If input data is an xray.DataArray,
        data_i is returned as an xray.DataArray, otherwise as
        an ndarray.
    """

    # Maximum number of dimensions handled by this code
    nmax = 5
    ndim = data.ndim

    if ndim > 5:
        raise ValueError('Input data has too many dimensions. Max 5-D.')

    if isinstance(data, xray.DataArray):
        i_DataArray = True
        data = data.copy()
        coords, attrs, name = xr.meta(data)
        title = 'Pressure-level data interpolated onto new pressure grid'
        attrs = utils.odict_insert(attrs, 'title', title, pos=0)
        pname = get_coord(data, 'plev', 'name')
        plev_in = get_coord(data, 'plev')
        coords[pname] = xray.DataArray(plev_new, coords={pname : plev_new},
            attrs=data.coords[pname].attrs)
    else:
        i_DataArray = False

    # Make sure pressure units are consistent
    if plev_new.min() < plev_in.min() or plev_new.max() > plev_in.max():
        raise ValueError('Output pressure levels are not contained '
            'within input pressure levels.  Check units on each.')

    # Add singleton dimensions for looping, if necessary
    for i in range(ndim, nmax):
        data = np.expand_dims(data, axis=0)

    # Make sure pdim is indexing from end
    pdim_in = pdim
    if pdim > 0:
        pdim = pdim - nmax

    # Iterate over all other dimensions
    dims = list(data.shape)
    dims[pdim] = len(plev_new)
    data_i = np.nan*np.ones(dims, dtype=float)
    dims.pop(pdim)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                for m in range(dims[3]):
                    if pdim == -3:
                        sub = data[i,j,:,k,m]
                        view = data_i[i,j,:,k,m]
                    elif pdim == -2:
                        sub = data[i,j,k,:,m]
                        view = data_i[i,j,k,:,m]
                    elif pdim == -1:
                        sub = data[i,j,k,m,:]
                        view = data_i[i,j,k,m,:]
                    else:
                        raise ValueError('Invalid p dimension ' + str(pdim_in))

                    vals_i = interp.interp1d(plev_in, sub, kind=kind)(plev_new)
                    view[:] = vals_i

    # Collapse any additional dimensions that were added
    for i in range(ndim, data_i.ndim):
        data_i = data_i[0]

    # Pack data_s into an xray.DataArray if input was in that form
    if i_DataArray:
        data_i = xray.DataArray(data_i, name=name, coords=coords,
                                attrs=attrs)

    return data_i


# ----------------------------------------------------------------------
def int_pres(data, plev=None, pdim=-3, pmin=0, pmax=1e6):
    """Return the mass-weighted vertical integral of the data.

    Parameters
    ----------
    data : xray.DataArray or ndarray
        Data to be integrated, on pressure levels.
    plev : ndarray, optional
        Vertical pressure levels in Pascals.  Only used if data
        is an ndarray.  If data is a DataArray, plev is extracted
        from data and converted to Pa if necessary.
    pdim : int, optional
        Dimension of vertical pressure levels in data.
    pmin, pmax : float, optional
        Lower and upper bounds (inclusive) of pressure levels (Pa)
        to include in integration.

    Returns
    -------
    data_int : xray.DataArray or ndarray
        Mass-weighted vertical integral of data from pmin to pmax.
    """

    if isinstance(data, xray.DataArray):
        i_DataArray = True
        data = data.copy()
        coords, _, name = xr.meta(data)
        attrs = collections.OrderedDict()
        title = 'Vertically integrated by dp/g'
        attrs['title'] = title
        if 'long_name' in data.attrs.keys():
            attrs['long_name'] = data.attrs['long_name']
        if 'units' in data.attrs.keys():
            attrs['units'] = '(' + data.attrs['units'] + ') * kg'
        pname = get_coord(data, 'plev', 'name')
        del(coords[pname])
        if plev is None:
            # -- Make sure pressure levels are in Pa
            plev = get_coord(data, 'plev')
            plev = pres_convert(plev, data[pname].units, 'Pa')
        data[pname].values = plev
    else:
        i_DataArray = False
        # Pack into DataArray to easily extract pressure level subset
        pname = 'plev'
        coords = xr.coords_init(data)
        coords = xr.coords_assign(coords, pdim, pname, plev)
        data = xray.DataArray(data, coords=coords)

    # Extract subset and integrate
    data = subset(data, pname, pmin, pmax)
    vals_int = nantrapz(data.values, data[pname].values, axis=pdim)
    vals_int /= constants.g.values

    if utils.strictly_decreasing(plev):
        vals_int = -vals_int

    if i_DataArray:
        data_int = xray.DataArray(vals_int, name=name, coords=coords,
                                  attrs=attrs)
    else:
        data_int = vals_int

    return data_int




# LAT-LON GEO
# def average_over_country():
#    """Return the data field averaged over a country."""
