import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat

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

u_s, ind_s = dat.near_surface(u, pdim=-3, return_inds=True)

m = 0
plt.figure(figsize=(7,10))
plt.subplot(311)
ap.pcolor_latlon(u[m,0], cmap='jet')
plt.subplot(312)
ap.pcolor_latlon(u_s[m], cmap='jet')
plt.subplot(313)
ap.pcolor_latlon(ind_s[m], lat, lon, cmap='jet')
