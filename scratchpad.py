# Naming conventions for importing standard scientific modules:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Additional modules useful for atmospheric science
from mpl_toolkits.basemap import Basemap
import xray

def unpack(dataset, missing_name=u'missing_value', offset_name=u'add_offset',
    scale_name=u'scale_factor', debug=False):
    '''Unpack data from netCDF dataset, converting compressed int data to
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

# Read data from netcdf and unpack
with xray.open_dataset('data/uwnd.mon.mean.nc', decode_cf=False) as ds1:
    ds1 = unpack(ds1)
    lat = ds1['lat'].values.astype(np.float64)
    lon = ds1['lon'].values.astype(np.float64)
    lev = ds1['level'].values
    time = ds1['time'].values
    u = ds1['uwnd'].values.astype(np.float64)

with xray.open_dataset('data/vwnd.mon.mean.nc', decode_cf=False) as ds2:
    ds2 = unpack(ds2)
    v = ds2['vwnd'].values.astype(np.float64)

with xray.open_dataset('data/air.mon.mean.nc', decode_cf=False) as ds3:
    ds3 = unpack(ds3)
    T = ds3['air'].values.astype(np.float64)

with xray.open_dataset('data/pres.sfc.mon.mean.nc', decode_cf=False) as ds4:
    ds4 = unpack(ds4)
    ps = ds4['pres'].values.astype(np.float64)


# Remove Jan-Jun of 2015
tlast = -7
time = time[:tlast]
u = u[:tlast]
v = v[:tlast]
T = T[:tlast]
ps = ps[:tlast]

# Take annual mean and climatology over all years
ubar = u.mean(axis=0)
vbar = v.mean(axis=0)
Tbar = T.mean(axis=0)
psbar = ps.mean(axis=0)

# Reshape to add month as a dimension
u1, v1, T1, ps1 = u, v, T, ps
(nt,nz,ny,nx) = u.shape
nmon = 12
nyear = nt / 12
shape = (nyear, nmon, nz, ny, nx)
shape2 = (nyear, nmon, ny, nx)
u = u.reshape(shape, order='C')
v = v.reshape(shape, order='C')
T = T.reshape(shape, order='C')
ps = ps.reshape(shape2, order='C')

# Check that reshaping worked properly
m, y, k = 11, 20, 10
data1 = u1[y*12 + m, k]
data2 = u[y, m, k]
print(np.array_equal(data1,data2))

# Plot data to check
xi, yi = np.meshgrid(lon, lat)
plt.figure()
plt.subplot(211)
plt.pcolormesh(xi, yi, data1, cmap='jet')
plt.colorbar()
plt.subplot(212)
plt.pcolormesh(xi,yi, data2, cmap='jet')
plt.colorbar()


# Create a new dataset with the climatology
outfile = 'data/ncep2_climatology.nc'
title = '1979-2014 Annual Mean Climatology from NCEP-DOE AMIP-II Reanalysis'
ds = xray.Dataset()
ds.attrs['title'] = title
ds.coords['lat'] = ('y', lat)
ds.coords['lon'] = ('x', lon)
ds.coords['lev'] = ('z', lev)
ds['u'] = ( ('z', 'y', 'x'), ubar)
ds['v'] = ( ('z', 'y', 'x'), vbar)
ds['T'] = ( ('z', 'y', 'x'), Tbar)
ds['ps'] = ( ('y', 'x'), psbar)

# Save to netCDF file
ds.to_netcdf(outfile, mode='w')
