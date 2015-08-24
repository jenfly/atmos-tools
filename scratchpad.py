# Standard scientific modules:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import xray

# My modules:
import xray_tools as xr

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# Read data from netcdf and unpack
with xray.open_dataset('data/uwnd.mon.mean.nc', decode_cf=False) as ds1:
    xr.ds_print(ds1)
    ds1 = xr.ds_unpack(ds1)
    lat = ds1['lat'].values.astype(np.float64)
    lon = ds1['lon'].values.astype(np.float64)
    lev = ds1['level'].values
    time = ds1['time'].values
    u = ds1['uwnd'].values.astype(np.float64)

with xray.open_dataset('data/vwnd.mon.mean.nc', decode_cf=False) as ds2:
    xr.ds_print(ds2)
    ds2 = xr.ds_unpack(ds2)
    v = ds2['vwnd'].values.astype(np.float64)

with xray.open_dataset('data/air.mon.mean.nc', decode_cf=False) as ds3:
    xr.ds_print(ds3)
    ds3 = xr.ds_unpack(ds3)
    T = ds3['air'].values.astype(np.float64)

with xray.open_dataset('data/pres.sfc.mon.mean.nc', decode_cf=False) as ds4:
    xr.ds_print(ds4)
    ds4 = xr.ds_unpack(ds4)
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

# Create a new dataset with the reshaped monthly data and save to netcdf
outfile = 'data/ncep2_monthly_'
title = '1979-2014 Monthly Means from NCEP-DOE AMIP-II Reanalysis'
ds = xray.Dataset()
ds.attrs['title'] = title
ds.coords['lat'] = ('lat', lat)
ds.coords['lon'] = ('lon', lon)
ds.coords['lev'] = ('lev', lev)
ds.coords['mon'] = ('mon', np.arange(1,13))
ds.coords['yr'] = ('yr', np.arange(1979,2015))
ds['u'] = ( ('yr', 'mon', 'lev', 'lat', 'lon'), u)
ds.to_netcdf(outfile + 'u.nc', mode='w')
ds = ds.drop('u')
ds['v'] = ( ('yr', 'mon', 'lev', 'lat', 'lon'), v)
ds.to_netcdf(outfile + 'v.nc', mode='w')
ds = ds.drop('v')
ds['T'] = ( ('yr', 'mon', 'lev', 'lat', 'lon'), T)
ds.to_netcdf(outfile + 'T.nc', mode='w')
ds = ds.drop('T')
ds['ps'] = ( ('yr', 'mon', 'lat', 'lon'), ps)
ds.to_netcdf(outfile + 'ps.nc', mode='w')

# Monthly climatology
outfile = 'data/ncep2_climatology_monthly.nc'
title = '1979-2014 Monthly Mean Climatology from NCEP-DOE AMIP-II Reanalysis'
ds.attrs['title'] = title
ds = ds.drop('yr')
ds['u'] = ( ('mon', 'lev', 'lat', 'lon'), u.mean(axis=0))
ds['v'] = ( ('mon', 'lev', 'lat', 'lon'), v.mean(axis=0))
ds['T'] = ( ('mon', 'lev', 'lat', 'lon'), T.mean(axis=0))
ds['ps'] = ( ('mon', 'lat', 'lon'), ps.mean(axis=0))
ds.to_netcdf(outfile, mode='w')

# Annual mean climatology
outfile = 'data/ncep2_climatology_ann.nc'
title = '1979-2014 Annual Mean Climatology from NCEP-DOE AMIP-II Reanalysis'
ds = xray.Dataset()
ds.attrs['title'] = title
ds.coords['lat'] = ('lat', lat)
ds.coords['lon'] = ('lon', lon)
ds.coords['lev'] = ('lev', lev)
ds['u'] = ( ('lev', 'lat', 'lon'), ubar)
ds['v'] = ( ('lev', 'lat', 'lon'), vbar)
ds['T'] = ( ('lev', 'lat', 'lon'), Tbar)
ds['ps'] = ( ('lat', 'lon'), psbar)
ds.to_netcdf(outfile, mode='w')


# ----------------------------------------------------------------------
# Read monthly mean climatologies and do some test calcs
# ----------------------------------------------------------------------

filename = 'data/ncep2_climatology_monthly.nc'
with xray.open_dataset(filename) as ds:
    lat = ds['lat'].values
    lon = ds['lon'].values
    lev = ds['lev'].values
    mon = ds['mon'].values
    u = ds['u'].values
    v = ds['v'].values
    T = ds['T'].values
    ps = ds['ps'].values

xi, yi = np.meshgrid(lon, lat)
k, mon = 9, 7
plt.figure()
plt.pcolormesh(xi, yi, u[mon-1, k])
plt.colorbar()
