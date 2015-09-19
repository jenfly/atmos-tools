import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
from atmos.data import near_surface

# ----------------------------------------------------------------------
# Read monthly mean climatologies and do some test calcs
# ----------------------------------------------------------------------

filename = 'data/more/ncep2_climatology_monthly.nc'
ds = dat.ncload(filename)
lat = ds['lat'].values
lon = ds['lon'].values
plev = ds['plev'].values
mon = ds['mon'].values
u = ds['u']
v = ds['v']
T = ds['T']
ps = ds['ps']

topo = dat.get_ps_clim(lat, lon) / 100
topo.units = 'hPa'

# ----------------------------------------------------------------------
# Correct for topography

u_orig = u
u = dat.correct_for_topography(u_orig, topo)

# ----------------------------------------------------------------------
# Near-surface data

# DataArray
u_s, ind_s = near_surface(u, pdim=-3, return_inds=True)

# ndarray
u_s2 = near_surface(u.values, pdim=-3, return_inds=False)

m = 0
plt.figure(figsize=(12,10))
plt.subplot(221)
ap.pcolor_latlon(u[m,0], cmap='jet')
plt.subplot(222)
ap.pcolor_latlon(u_s[m], cmap='jet')
plt.subplot(223)
ap.pcolor_latlon(u_s2[m], lat, lon, cmap='jet')
plt.subplot(224)
ap.pcolor_latlon(ind_s[m], lat, lon, cmap='jet')
