# Standard scientific modules:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import xray
from datetime import datetime

# My modules:
import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
import atmos.variables as av
from atmos.utils import print_if
from atmos.constants import const as constants

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

# ======================================================================
# OpenDAP

# Use xray to open an HDF OpenDAP file!

url = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA_MONTHLY/'
    'MAIMCPASM.5.2.0/1979/MERRA100.prod.assim.instM_3d_asm_Cp.197901.hdf')

ds = xray.open_dataset(url)
T = ds['T']
ps = ds['PS']
q = ds['QV']
plev = dat.get_plev(T, units='Pa')
lat, lon = dat.get_lat(ps), dat.get_lon(ps)
p0 = 1e5

theta = av.potential_temp(T, plev, p0)
theta_e = av.equiv_potential_temp(T, plev, q, p0)

# t, k = 0, 6
# t, k = 0, 22
t, k = 0, 14
pstr = '%d hPa' % (plev[k]/100)
plt.figure(figsize=(7,10))
plt.subplot(311)
ap.pcolor_latlon(T[t,k])
plt.title('Temperature ' + pstr)
plt.subplot(312)
ap.pcolor_latlon(theta[t,k])
plt.title('Potential Temperature ' + pstr)
plt.subplot(313)
ap.pcolor_latlon(theta_e[t,k])
plt.title('Equiv Potential Temperature ' + pstr)

ps.dims
plt.figure()
ap.pcolor_latlon(ps,cmap='hot')


# ----------------------------------------------------------------------
# Averaging over box
#lon1, lon2 = 70, 90
#lat1, lat2 = 10, 25
lon1, lon2 = 15, 30
lat1, lat2 = -50, -15

avg1 = dat.mean_over_geobox(T, lat1, lat2, lon1, lon2, area_wtd=False)
avg2 = dat.mean_over_geobox(T, lat1, lat2, lon1, lon2, area_wtd=True)
avg3 = dat.mean_over_geobox(T, lat1, lat2, lon1, lon2, area_wtd=True,
                            land_only=True)

t, k = 0, 6
plt.figure()
ap.pcolor_latlon(T[t,k], axlims=(lat1,lat2,lon1,lon2), cmap='jet')
plt.clim(270, 300)

print(avg1[t, k].values)
print(avg2[t, k].values)
print(avg3[t, k].values)

# ----------------------------------------------------------------------
# Masking ocean

T_land = dat.mask_oceans(T, inlands=True)

t, k = 0, 10
plt.figure()
ap.pcolor_latlon(T_land[t,k], cmap='jet')
