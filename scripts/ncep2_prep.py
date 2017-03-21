# Standard scientific modules:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import xarray as xray

# My modules:
import atmos as atm

# ----------------------------------------------------------------------
# Read data from netcdf and unpack

miss = u'missing_value'
offset = u'add_offset'
scale = u'scale_factor'

def filename(var):
    datadir = '~/datastore/atmos-tools/'
    filen = datadir + var + '.mon.mean.nc'
    print('Loading ' + filen)
    return filen

ds = atm.ncload(filename('uwnd'), verbose=True, unpack=True,
    missing_name=miss, offset_name=offset, scale_name=scale, decode_cf=False)
ds.rename({'uwnd': 'u'}, inplace=True)

ds2 = atm.ncload(filename('vwnd'), verbose=True, unpack=True,
    missing_name=miss, offset_name=offset, scale_name=scale, decode_cf=False)
ds['v'] = ds2['vwnd']

ds2 = atm.ncload(filename('air'), verbose=True, unpack=True,
    missing_name=miss, offset_name=offset, scale_name=scale, decode_cf=False)
ds['T'] = ds2['air']

ds2 = atm.ncload(filename('pres.sfc'), verbose=True, unpack=True,
    missing_name=miss, offset_name=offset, scale_name=scale, decode_cf=False)
ds['ps'] = ds2['pres']

ds.rename({'level' : 'plev'}, inplace=True)
lat = ds['lat'].values.astype(np.float64)
lon = ds['lon'].values.astype(np.float64)
plev = ds['plev'].values.astype(np.float64)
time = ds['time'].values

# Replace float32 with float64 so that everything is in float64
ds['lat'].values = lat
ds['lon'].values = lon
ds['plev'].values = plev

# ----------------------------------------------------------------------
# Data processing

# Remove attributes related to data compression
data_vars = ['u', 'v', 'T', 'ps']
att_remove = [miss, offset, scale, u'valid_range']

for var in data_vars:
    print(var)
    for att in att_remove:
        if att in ds[var].attrs:
            print('  ' + att)
            del(ds[var].attrs[att])

# Unpack into numpy arrays
u = ds['u'].values
v = ds['v'].values
T = ds['T'].values
ps = ds['ps'].values

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

# ----------------------------------------------------------------------
# Save reshaped monthly data to netcdf, one file per data variable

outfile = 'data/more/ncep2_monthly_'
title = '1979-2014 Monthly NCEP/DOE Renalysis 2'
ds1 = ds.drop(['time', 'nbnds'])
ds1.attrs['title'] = title
ds1.coords['mon'] = ('mon', np.arange(1,13))
ds1.coords['yr'] = ('yr', np.arange(1979,2015))

ds1['u'] = ( ('yr', 'mon', 'plev', 'lat', 'lon'), u)
ds1['v'] = ( ('yr', 'mon', 'plev', 'lat', 'lon'), v)
ds1['T'] = ( ('yr', 'mon', 'plev', 'lat', 'lon'), T)
ds1['ps'] = ( ('yr', 'mon', 'lat', 'lon'), ps)

for var in ['u', 'v', 'T', 'ps']:
    ds1[var].attrs = ds[var].attrs
    ds1[var].attrs[u'actual_range'] = [ds1[var].values.min(),
        ds1[var].values.max()]

ds2 = ds1.drop(['v', 'T', 'ps'])
ds2.to_netcdf(outfile + 'u.nc', mode='w')

ds2 = ds1.drop(['u', 'T', 'ps'])
ds2.to_netcdf(outfile + 'v.nc', mode='w')

ds2 = ds1.drop(['u', 'v', 'ps'])
ds2.to_netcdf(outfile + 'T.nc', mode='w')

ds2 = ds1.drop(['u', 'v', 'T'])
ds2.to_netcdf(outfile + 'ps.nc', mode='w')

# ----------------------------------------------------------------------
# Save monthly climatology to netcdf

outfile = 'data/more/ncep2_climatology_monthly.nc'
title = '1979-2014 Monthly Mean Climatology NCEP/DOE Reanalysis 2'
ds1 = ds.drop(['time', 'nbnds'])
ds1.attrs['title'] = title
ds1.coords['mon'] = ('mon', np.arange(1,13))

ds1['u'] = ( ('mon', 'plev', 'lat', 'lon'), u.mean(axis=0))
ds1['v'] = ( ('mon', 'plev', 'lat', 'lon'), v.mean(axis=0))
ds1['T'] = ( ('mon', 'plev', 'lat', 'lon'), T.mean(axis=0))
ds1['ps'] = ( ('mon', 'lat', 'lon'), ps.mean(axis=0))

for var in ['u', 'v', 'T', 'ps']:
    ds1[var].attrs = ds[var].attrs
    ds1[var].attrs[u'actual_range'] = [ds1[var].values.min(),
        ds1[var].values.max()]

ds1.to_netcdf(outfile, mode='w')

# ----------------------------------------------------------------------
# Save annual mean climatology to netcdf

outfile = 'data/ncep2_climatology_ann.nc'
title = '1979-2014 Annual Mean Climatology NCEP/DOE Reanalysis 2'
ds1 = ds.drop(['time', 'nbnds'])
ds1.attrs['title'] = title

ds1['u'] = ( ('plev', 'lat', 'lon'), ubar)
ds1['v'] = ( ('plev', 'lat', 'lon'), vbar)
ds1['T'] = ( ('plev', 'lat', 'lon'), Tbar)
ds1['ps'] = ( ('lat', 'lon'), psbar)

for var in ['u', 'v', 'T', 'ps']:
    ds1[var].attrs = ds[var].attrs
    ds1[var].attrs[u'actual_range'] = [ds1[var].values.min(),
        ds1[var].values.max()]
    s = ds1[var].attrs[u'long_name']
    ds1[var].attrs[u'long_name'] = s.replace('Monthly ', '')

ds1.to_netcdf(outfile, mode='w')

# ----------------------------------------------------------------------
# Save surface pressure climatology to topo file

outfile = 'data/topo/ncep2_ps.nc'
ds2 = ds1.drop(['u', 'v', 'T', 'plev'])
ds2.to_netcdf(outfile, mode='w')
