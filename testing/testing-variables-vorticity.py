import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
from atmos.constants import const as constants
from atmos.data import get_coord
from atmos.variables import vorticity, rossby_num


# ----------------------------------------------------------------------
# Read data
url = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA_MONTHLY/'
    'MAIMCPASM.5.2.0/1979/MERRA100.prod.assim.instM_3d_asm_Cp.197907.hdf')

ds = xray.open_dataset(url)
#T = ds['T']
#ps = ds['PS']
u = ds['U']
v = ds['V']
#q = ds['QV']
lat = get_coord(v, 'lat')
lon = get_coord(v, 'lon')
plev = get_coord(v, 'plev')

topo = dat.get_ps_clim(lat, lon) / 100
topo.units = 'hPa'

# ----------------------------------------------------------------------
# Relative and absolute vorticity

# DataArray
rel_vort, abs_vort, f = vorticity(u, v)

# ndarray
rel_vort2, abs_vort2, f2 = vorticity(u.values, v.values, lat, lon)


t, k = 0, 22
plt.figure(figsize=(12,8))
plt.subplot(221)
ap.pcolor_latlon(rel_vort[t,k])
plt.subplot(222)
ap.pcolor_latlon(abs_vort[t,k])
plt.subplot(223)
ap.pcolor_latlon(rel_vort2[t,k], lat, lon)
plt.subplot(224)
ap.pcolor_latlon(abs_vort2[t,k], lat, lon)

# ----------------------------------------------------------------------
# Rossby number

Ro = rossby_num(u, v)
Ro2 = rossby_num(u.values, v.values, lat, lon)

t, k = 0, 22
lon1, lon2 = 60, 100
lat1, lat2 = 10, 50
cmax = 1.5
axlims = (lat1, lat2, lon1, lon2)
plt.figure(figsize=(7,8))
plt.subplot(211)
ap.pcolor_latlon(Ro[t,k], axlims=axlims)
plt.clim(-cmax, cmax)
plt.subplot(212)
ap.pcolor_latlon(Ro2[t,k], lat, lon, axlims=axlims)
plt.clim(-cmax, cmax)
