import numpy as np
import xray

# ----------------------------------------------------------------------
# Utility functions for working with xray datasets and netCDF files
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def odict_print(od, indent=2, width=20):
    '''Pretty print the contents of an ordered dictionary.'''
    for key in od:
        s = ' ' * indent + key
        print(s.ljust(width) + str(od[key]))

# ----------------------------------------------------------------------
def ds_print(ds, indent=2, width=20):
    '''Print metadata for xray dataset and each coordinate and data variable.'''

    line = '-' * 60

    # Metadata for dataset as a whole
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
def ncdisp(filename, details=True, decode_cf=False, indent=2, width=20):
    with xray.open_dataset(filename, decode_cf=decode_cf) as ds:
        if details:
            ds_print(ds, indent, width)
        else:
            print(ds)

# ----------------------------------------------------------------------
def ds_unpack(dataset, missing_name=u'missing_value', offset_name=u'add_offset',
    scale_name=u'scale_factor', debug=False):
    '''Unpack data from netCDF file, converting compressed int data to
    floats and missing values to NaN'''
    ds = dataset
    for var in ds.data_vars:
        print(var)
        vals = ds[var].values
        attrs = ds[var].attrs
        if debug: print(attrs)

        # Flag missing values for further processing
        if missing_name in attrs:
            missing_val = attrs[missing_name]
            imissing = vals == missing_val
            print('missing_val ', str(missing_val))
        else:
            imissing = []

        # Get offset and scaling factors, if any
        if offset_name in attrs:
            offset_val = attrs[offset_name]
            print('offset val ', str(offset_val))
        else:
            offset_val = 0.0
        if scale_name in attrs:
            scale_val = attrs[scale_name]
            print('scale val ', str(scale_val))
        else:
            scale_val = 1.0

        # Convert from int to float with the offset and scaling
        vals = vals * scale_val + offset_val

        # Replace missing values with NaN
        vals[imissing] = np.nan

        # Replace the values in dataset with the converted ones
        ds[var].values = vals

    return ds
