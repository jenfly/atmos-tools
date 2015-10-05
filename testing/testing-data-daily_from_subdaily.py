import xray
import numpy as np
import matplotlib.pyplot as plt
import atmos as atm
from atmos import daily_from_subdaily

datadir = '/home/jwalker/eady/datastore/'
#datadir = '/home/jennifer/datastore/'

# ----------------------------------------------------------------------
# MERRA Daily
filename = datadir + 'merra/daily/merra_u200_198601.nc'
ds = atm.ncload(filename)
u = ds['U']
lat = atm.get_coord(u, 'lat')
lon = atm.get_coord(u, 'lon')

# Number of time points per day
n = 8

# Daily mean
u_split = atm.split_timedim(u, n, time0_name='day')
u_new = daily_from_subdaily(u, n, dayvals=np.arange(1,32))
print(np.array_equal(u_new, u_split.mean(axis=1)))

# ndarray version
u_new2 = daily_from_subdaily(u.values, n)
print(np.array_equal(u_new, u_new2))

# Sub-sample version
i = 2
u_new3 = daily_from_subdaily(u, n, method=i)
print(np.array_equal(u_split[:,i], u_new3))


# Plot data to check
d = 5
plt.figure()
plt.subplot(211)
atm.pcolor_latlon(u_new[d], cmap='jet')
plt.subplot(212)
atm.pcolor_latlon(u_new.mean(axis=0), cmap='jet')


