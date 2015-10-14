import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos as atm

# ----------------------------------------------------------------------
# Read monthly mean climatologies and do some test calcs
# ----------------------------------------------------------------------

filename = 'data/more/ncep2_climatology_monthly.nc'
ds = atm.ncload(filename)
lat = ds['lat'].values
lon = ds['lon'].values
plev = ds['plev'].values
mon = ds['mon'].values
u = ds['u']
v = ds['v']
T = ds['T']
ps = ds['ps']

topo = atm.get_ps_clim(lat, lon) / 100
topo.units = 'hPa'

# ----------------------------------------------------------------------
# Correct for topography

u_orig = u
u = atm.correct_for_topography(u_orig, topo)

# ----------------------------------------------------------------------
# Zonal mean zonal wind
season='jjas'
lon1, lon2 = 60, 100
cint = 5
months = atm.season_months(season)

uplot = atm.subset(u, 'lon', lon1, lon2, 'mon', months)
uplot = uplot.mean(['lon', 'mon'])

ps_plot = atm.subset(topo, 'lon', lon1, lon2)
ps_plot = ps_plot.mean('lon')

plt.figure()
cs = atm.contour_latpres(uplot, clev=cint, topo=ps_plot)
clev = atm.clevels(uplot, cint, omitzero=True)
plt.clabel(cs, clev[::2], fmt='%02d')

plt.figure()
atm.contourf_latpres(uplot,clev=cint, topo=ps_plot)
