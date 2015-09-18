"""Testing  mask_oceans"""

import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
from atmos.data import get_coord, mask_oceans

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
# mask_oceans

# DataArray input
ps_land = mask_oceans(ps)

# ndarray input
ps_land2 = mask_oceans(ps, lat, lon)

t = 0
cmap='jet'

plt.figure(figsize=(7,10))
plt.subplot(311)
ap.pcolor_latlon(ps[t], cmap=cmap)
plt.subplot(312)
ap.pcolor_latlon(ps_land[t], cmap=cmap)
plt.subplot(313)
ap.pcolor_latlon(ps_land2[t], lat, lon, cmap=cmap)
