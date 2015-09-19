import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
from atmos.constants import const as constants
from atmos.data import get_coord, int_pres

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
# Integrated vertically dp/g

# DataArray
u_int = int_pres(u, pdim=-3)

# ndarray
u_int2 = int_pres(u.values, plev*100, pdim=-3)

p0=1e5
g = constants.g.values
scale = g/p0

m = 7
k = 5
cint=2

plt.figure(figsize=(7,10))
plt.subplot(311)
ap.contourf_latlon(u[m,k], clev=cint)
plt.subplot(312)
ap.contourf_latlon(scale*u_int[m], clev=cint)
plt.subplot(313)
ap.contourf_latlon(scale*u_int2[m], lat, lon, clev=cint)

# ----------------------------------------------------------------------
# Integrate over subset
# pmin = 400e2
# pmax = 600e2
# m, k = 3, 5
pmin, pmax = 600e2, 1000e2
m, k = 3, 2
scale = g/(pmax-pmin)
cint=1

u_int = int_pres(u, pdim=-3, pmin=pmin, pmax=pmax)
u_int2 = int_pres(u.values, plev*100, pdim=-3, pmin=pmin, pmax=pmax)

plt.figure(figsize=(7,10))
plt.subplot(311)
ap.contourf_latlon(u[m,k], clev=cint)
plt.subplot(312)
ap.contourf_latlon(scale*u_int[m], clev=cint)
plt.subplot(313)
ap.contourf_latlon(scale*u_int2[m], lat, lon, clev=cint)
