"""Testing  interp_latlon"""

import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
from atmos.data import get_coord, interp_latlon

# ----------------------------------------------------------------------
# NCEP2
# ----------------------------------------------------------------------

filename = 'data/more/ncep2_climatology_monthly.nc'
ds = dat.ncload(filename)
mon = ds['mon'].values
u = ds['u']
v = ds['v']
T = ds['T']
ps = ds['ps']
lat = get_coord(ps, 'lat')
lon = get_coord(ps, 'lon')
plev = get_coord(u, 'plev')

# ----------------------------------------------------------------------
# interp_latlon

lat_new = np.arange(-90, 90, 1.)
lon_new = np.arange(0, 360, 1.)

# DataArray input
ps_i = interp_latlon(ps, lat_new, lon_new)

# ndarray input
ps_i2 = interp_latlon(ps.values, lat_new, lon_new, lat, lon)

t = 0
cmap='jet'

plt.figure(figsize=(7,10))
plt.subplot(311)
ap.pcolor_latlon(ps[t], cmap=cmap)
plt.subplot(312)
ap.pcolor_latlon(ps_i[t], cmap=cmap)
plt.subplot(313)
ap.pcolor_latlon(ps_i2[t], lat_new, lon_new, cmap=cmap)
