import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
from atmos.data import get_coord, interp_plevels

# ----------------------------------------------------------------------
# Read monthly mean climatologies and do some test calcs
# ----------------------------------------------------------------------

filename = 'data/more/ncep2_climatology_monthly.nc'
ds = dat.ncload(filename)
u = ds['u']
v = ds['v']
T = ds['T']
ps = ds['ps']
lat = get_coord(u, 'lat')
lon = get_coord(u, 'lon')
plev = get_coord(u, 'plev')
mon = ds['mon'].values

topo = dat.get_ps_clim(lat, lon) / 100
topo.units = 'hPa'

# ----------------------------------------------------------------------
# Correct for topography

u_orig = u
u = dat.correct_for_topography(u_orig, topo)

# ----------------------------------------------------------------------
# Interpolate onto new pressure grid

plev_new = np.arange(1000,0,-25.)

# DataArray
u_i = interp_plevels(u, plev_new, pdim=-3)

# ndarray
u_i2 = interp_plevels(u, plev_new, plev, pdim=-3)

m, i, j = 0, 35, 70
cint=5

plt.figure(figsize=(7,8))
plt.subplot(211)
plt.plot(u[m,:,i,j], plev, 'b.-')
plt.plot(u_i[m,:,i,j], plev_new, 'r*')
plt.gca().invert_yaxis()
plt.subplot(212)
plt.plot(u[m,:,i,j], plev, 'b.-')
plt.plot(u_i2[m,:,i,j], plev_new, 'r*')
plt.gca().invert_yaxis()
plt.draw()


plt.figure(figsize=(7,10))
plt.subplot(311)
ap.contour_latpres(u[m,:,:,j], clev=cint)
plt.subplot(312)
ap.contour_latpres(u_i[m,:,:,j], clev=cint)
plt.subplot(313)
ap.contour_latpres(u_i2[m,:,:,j], lat, plev, clev=cint)
