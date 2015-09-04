# Standard scientific modules:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import xray

# My modules:
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

topo = dat.get_ps_clim(lat, lon)

# ----------------------------------------------------------------------
# Correct for topography

ucor = dat.correct_for_topography(u, topo)

m, k = 3, 1

plt.figure(figsize=(7,8))
plt.subplot(211)
ap.pcolor_latlon(u[m,k], cmap='jet')
plt.subplot(212)
ap.pcolor_latlon(ucor[m,k], cmap='jet')

# ----------------------------------------------------------------------

# Zonal mean zonal wind
season='jjas'
lon1, lon2 = 60, 100
pmax = 1100
cint = 5

imon = utils.season_months(season)
ilon = (lon >= lon1) & (lon <= lon2)

uplot = ucor[imon]
uplot = uplot[:,:,:,ilon]
uplot = uplot.mean(axis=3).mean(axis=0)

topo = dat.get_ps_clim(lat, lon) / 100
topo = topo[:,ilon]
topo = topo.mean(axis=1)

plt.figure()
ap.contour_latpres(uplot, clev=cint, topo=topo)

plt.figure()
ap.pcolor_latpres(uplot,topo=topo)
