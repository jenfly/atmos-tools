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

m, k = 3, 1
plt.figure(figsize=(7,8))
plt.subplot(211)
ap.pcolor_latlon(u_orig[m,k], cmap='jet')
plt.subplot(212)
ap.pcolor_latlon(u[m,k], cmap='jet')


# ----------------------------------------------------------------------
# Zonal mean zonal wind
season='jjas'
lon1, lon2 = 60, 100
cint = 5
months = utils.season_months(season)

uplot = dat.subset(u, 'lon', lon1, lon2, 'mon', months)
uplot = uplot.mean(['lon', 'mon'])

ps_plot = dat.subset(topo, 'lon', lon1, lon2)
ps_plot = ps_plot.mean('lon')

plt.figure()
ap.contour_latpres(uplot, clev=cint, topo=ps_plot)

plt.figure()
ap.contourf_latpres(uplot,clev=cint, topo=ps_plot)
