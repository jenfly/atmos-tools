# Standard scientific modules:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import xray

# My modules:
import atmos.utils as utils
import atmos.xrhelper as xr
import atmos.plots as ap

# ----------------------------------------------------------------------
# Read monthly mean climatologies and do some test calcs
# ----------------------------------------------------------------------

filename = 'data/more/ncep2_climatology_monthly.nc'
ds = xr.ncload(filename)
lat = ds['lat'].values
lon = ds['lon'].values
lev = ds['lev'].values
mon = ds['mon'].values
u = ds['u'].values
v = ds['v'].values
T = ds['T'].values
ps = ds['ps'].values

xi, yi = np.meshgrid(lon, lat)
k, mon = 9, 7
uplot = u[mon-1, k]
plt.figure()
plt.contourf(xi, yi, uplot)
plt.colorbar()
plt.contour(xi,yi, uplot, np.arange(-40,60,10), colors='black')

# Zonal mean zonal wind
season='jjas'
lon1, lon2 = 60, 100
pmax = 1100
cint = 5

imon = utils.season_months(season)
ilon = (lon >= lon1) & (lon <= lon2)

uplot = u[imon]
uplot = uplot[:,:,:,ilon]
uplot = uplot.mean(axis=3).mean(axis=0)
#clev = ap.clevels(uplot, cint)

topo = ps.mean(axis=0) / 100
topo = topo[:,ilon].mean(axis=1)

plt.figure()
ap.contour_latpres(lat, lev, uplot, cint, topo=topo)


'''
ys,zs = np.meshgrid(lat, lev)
plt.figure()
plt.fill_between(lat,pmax,topo,color='black')
plt.contour(ys,zs,uplot,clev,colors='black')
plt.ylim(0, 1000)
plt.gca().invert_yaxis()
plt.xticks(np.arange(-90,90,30))
plt.xlabel('Latitude')
plt.ylabel('Pressure (mb)')
plt.draw()
'''
