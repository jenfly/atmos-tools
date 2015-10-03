import xray
import numpy as np
import matplotlib.pyplot as plt
import atmos as atm
from atmos import split_timedim

filename = '/home/jennifer/datastore/atmos-tools/uwnd.mon.mean.nc'
ds = atm.ncload(filename)
lat = ds['lat'].values.astype(np.float64)
lon = ds['lon'].values.astype(np.float64)
u = ds['uwnd']
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
