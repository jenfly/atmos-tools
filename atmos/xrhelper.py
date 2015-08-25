'''
Utility functions for working with xray datasets and netCDF files
'''

import numpy as np
import xray

# ----------------------------------------------------------------------
def odict_print(od, indent=2, width=20):
    '''Pretty print the contents of an ordered dictionary.'''
    for key in od:
        s = ' ' * indent + key
        print(s.ljust(width) + str(od[key]))

# ----------------------------------------------------------------------
def print_if(msg, verbose):
    if verbose:
        print(msg)

# ----------------------------------------------------------------------
def ds_print(ds, indent=2, width=20):
    '''
    Print metadata for xray dataset and for each coordinate and data variable.
    '''
    line = '-' * 60

    # Attributes for dataset as a whole
    print('\n' + line + '\nDATASET\n' + line)
    print(ds)

    # Coordinates attributes
    print('\n' + line + '\nCOORDINATES\n' + line)
    for coord in ds.coords:
        print(coord)
        odict_print(ds[coord].attrs, indent=indent, width=width)

    # Data variables attributes
    print('\n' + line + '\nDATA VARIABLES\n' + line)
    for var in ds.data_vars:
        print(var)
        odict_print(ds[var].attrs, indent=indent, width=width)

    print(line + '\n')

# ----------------------------------------------------------------------
def ncdisp(filename, verbose=True, decode_cf=False, indent=2, width=20):
    '''
    Display the attributes of data in a netcdf file.
    '''
    with xray.open_dataset(filename, decode_cf=decode_cf) as ds:
        if verbose:
            ds_print(ds, indent, width)
        else:
            print(ds)

# ----------------------------------------------------------------------
def ds_unpack(dataset, missing_name=u'missing_value', offset_name=u'add_offset',
              scale_name=u'scale_factor', verbose=False, dtype=np.float64):
    '''
    Unpack data from netcdf file as read with xray.open_dataset().

    Converts compressed int data to floats and missing values to NaN.
    '''
    ds = dataset
    for var in ds.data_vars:
        print(var)
        vals = ds[var].values
        attrs = ds[var].attrs
        if verbose:
            odict_print(attrs)

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
def ncread(filename, verbose=True, unpack=True, missing_name=u'missing_value',
           offset_name=u'add_offset', scale_name=u'scale_factor',
           decode_cf=False):
    '''
    Read data from netcdf file into xray dataset.

    If options are selected, unpacks from compressed form and/or replaces
    missing values with NaN.
    '''
    with xray.open_dataset(filename, decode_cf=decode_cf) as ds:
        if verbose:
            print('Reading file: ' + filename)
            ds_print(ds, indent=2, width=20)
        if unpack:
            print_if('Unpacking data', verbose)

            ds = ds_unpack(ds, verbose=verbose, missing_name=missing_name,
                offset_name=offset_name, scale_name=scale_name)

        # Use the load() function so that the dataset is available after
        # the file is closed
        ds.load()
        return ds
