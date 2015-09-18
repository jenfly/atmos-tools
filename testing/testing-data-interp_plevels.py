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
# Interpolate onto new pressure grid

plev_new = np.arange(1000,0,-50.)
u_i = dat.interp_plevels(u, plev_new, pdim=-3)

m, i, j = 0, 35, 70
cint=5

plt.figure()
plt.plot(u[m,:,i,j], plev, 'b.-')
plt.plot(u_i[m,:,i,j], plev_new, 'r*')
plt.gca().invert_yaxis()
plt.draw()


plt.figure(figsize=(7,8))
plt.subplot(211)
ap.contour_latpres(u[m,:,:,j], clev=cint)
plt.subplot(212)
ap.contour_latpres(u_i[m,:,:,j], clev=cint)
