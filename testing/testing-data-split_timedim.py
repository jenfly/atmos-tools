import xray
import numpy as np
import matplotlib.pyplot as plt
import atmos as atm
from atmos import split_timedim

datadir = '/home/jwalker/eady/datastore/'
#datadir = '/home/jennifer/datastore/'

# ----------------------------------------------------------------------
# NCEP2 Monthly
filename = datadir + 'atmos-tools/uwnd.mon.mean.nc'
ds = atm.ncload(filename)
u = ds['uwnd']
lat = atm.get_coord(u, 'lat')
lon = atm.get_coord(u, 'lon')

# Remove Jan-Jun of 2015
tlast = -7
u = u[:tlast]

unew = split_timedim(u.values, 12)
unew2 = split_timedim(u.values, 12, slowfast=False)

# Check that reshaping worked properly
m, y, k = 11, 20, 10
data1 = u[y*12 + m, k].values
data2 = unew[y, m, k]
data3 = unew2[m, y, k]
print(np.array_equal(data1,data2))
print(np.array_equal(data2, data3))

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
# DataArray version
years = np.arange(1979, 2015)
months = np.arange(1, 13)
unew = split_timedim(u, 12, time0_name='year', time0_vals=years,
                     time1_name='month', time1_vals=months)
unew2 = split_timedim(u, 12, slowfast=False)

# Check that reshaping worked properly
m, y, k = 11, 20, 10
data1 = u[y*12 + m, k]
data2 = unew[y, m, k]
data3 = unew2[m, y, k]
print(np.array_equal(data1,data2))
print(np.array_equal(data2, data3))

# Plot data to check
plt.figure()
plt.subplot(211)
atm.pcolor_latlon(data1, cmap='jet')
plt.subplot(212)
atm.pcolor_latlon(data2, cmap='jet')

# ----------------------------------------------------------------------
# MERRA Daily
filename = datadir + 'merra/daily/merra_u200_198601.nc'
ds = atm.ncload(filename)
u = ds['U']
lat = atm.get_coord(u, 'lat')
lon = atm.get_coord(u, 'lon')

# Number of time points per day
n = 8
days = np.arange(1, 32)
hrs = np.arange(0, 24, 3)
unew = split_timedim(u, n, time0_name='days', time0_vals=days,
                     time1_name='hours', time1_vals=hrs)
unew2 = split_timedim(u, n, slowfast=False)

# Check that reshaping worked properly
dy, hr, k = 1, 3, 0
data1 = u[dy*n + hr, k]
data2 = unew[dy, hr, k]
data3 = unew2[hr, dy, k]
print(np.array_equal(data1,data2))
print(np.array_equal(data2, data3))

# Plot data to check
plt.figure()
plt.subplot(211)
atm.pcolor_latlon(data1, cmap='jet')
plt.subplot(212)
atm.pcolor_latlon(data2, cmap='jet')


