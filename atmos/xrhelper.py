"""
Utility functions for xray DataArrays and Dataset.
"""

from __future__ import division
import numpy as np
import collections
import xray
from xray import Dataset

from atmos.utils import print_if, disptime
import atmos.utils as utils

# ======================================================================
# DATAARRAYS
# ======================================================================

# ----------------------------------------------------------------------
def meta(data):
    """Return the metadata from an xray.DataArray.

    Returns copies of the coordinates and attributes, rather than
    views, so that the output variables can be subsequently modified
    without accidentally changing the original DataArray.

    Parameters
    ----------
    data : xray.DataArray

    Returns
    -------
    coords : OrderedDict
    attrs : OrderedDict
    name : string

    Usage
    -----
    coords, attrs, name = meta(data)
    """

    # Create new variables and populate them to avoid inadvertent
    # views that could modify the originals later on
    attrs = collections.OrderedDict()
    for d in data.attrs:
        attrs[d] = data.attrs[d]

    coords = collections.OrderedDict()
    # Iterate in order of data.dims so that output is in the
    # same order as the data dimensions
    for key in data.dims:
        coords[key] = data.coords[key].copy(deep=True)

    return coords, attrs, data.name


# ----------------------------------------------------------------------
def coords_init(data):
    """Return OrderedDict of xray.DataArray-like coords for a numpy array.

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


# ----------------------------------------------------------------------
def subset(data, dim_name, lower_or_list, upper=None,
           dim_name2=None, lower_or_list2=None, upper2=None,
           incl_lower=True, incl_upper=True):
    """Extract a subset of a DataArray or Dataset along named dimensions.

    Returns a DataArray or Dataset sub extracted from input data,
    such that:
        sub[dim_name] >= lower_or_list & sub[dim_name] <= upper,
    OR  sub[dim_name] == lower_or_list (if lower_or_list is a list)
    And similarly for dim_name2, if included.

    Parameters
    ----------
    data : xray.DataArray or xray.Dataset
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
        sub : xray.DataArray or xray.Dataset
    """

    def subset_1dim(data, dim_name, lower_or_list, upper=None,
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

    sub = subset_1dim(data, dim_name, lower_or_list, upper, incl_lower,
                       incl_upper)

    if dim_name2 is not None:
        sub = subset_1dim(sub, dim_name2, lower_or_list2, upper2, incl_lower,
                           incl_upper)

    return sub


# ======================================================================
# DATASETS
# ======================================================================

# ----------------------------------------------------------------------
def ds_print(ds, indent=2, width=None):
    """Print attributes of xray dataset and each of its variables."""
    line = '-' * 60

    # Attributes for dataset as a whole
    print('\n' + line + '\nDATASET\n' + line)
    print(ds)

    # Coordinates attributes
    print('\n' + line + '\nCOORDINATES\n' + line)
    for coord in ds.coords:
        print(coord)
        utils.print_odict(ds[coord].attrs, indent=indent, width=width)

    # Data variables attributes
    print('\n' + line + '\nDATA VARIABLES\n' + line)
    for var in ds.data_vars:
        print(var)
        utils.print_odict(ds[var].attrs, indent=indent, width=width)

    print(line + '\n')


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
        print_if(attrs, verbose, printfunc=utils.print_odict)

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
def vars_to_dataset(*args):
    """Combine xray.DataArray variables into an xray.Dataset.

    Call Signatures
    ---------------
    vars_to_dataset(var1)
    vars_to_dataset(var1, var2)
    vars_to_dataset(var1, var2, var3)
    etc...

    Parameters
    ----------
    var1, var2, ... : xray.DataArrays

    Returns
    -------
    ds : xray.Dataset
    """

    # Get the first variable and initialize the dataset with it
    args = list(args)
    var = args.pop(0)
    ds = var.to_dataset()

    # Add the rest of the variables to the dataset
    for arg in args:
        ds[arg.name] = arg
    return ds
