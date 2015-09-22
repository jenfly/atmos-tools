import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
from atmos.constants import const as constants
from atmos.data import get_coord, divergence_spherical_2d


# ----------------------------------------------------------------------
# Read data
url = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA_MONTHLY/'
    'MAIMCPASM.5.2.0/1979/MERRA100.prod.assim.instM_3d_asm_Cp.197907.hdf')

ds = xray.open_dataset(url)
#T = ds['T']
#ps = ds['PS']
u = ds['U']
v = ds['V']
q = ds['QV']
lat = get_coord(u, 'lat')
lon = get_coord(u, 'lon')
plev = get_coord(u, 'plev')

topo = dat.get_ps_clim(lat, lon) / 100
topo.units = 'hPa'

# ----------------------------------------------------------------------
# Initial plots

ubar = u.mean(axis=-1)
vbar = v.mean(axis=-1)

plt.figure(figsize=(7,8))
plt.subplot(211)
ap.contour_latpres(ubar, clev=5)
plt.subplot(212)
ap.contour_latpres(vbar, clev=0.5)

t, k = 0, 4
plt.figure(figsize=(7,8))
plt.subplot(211)
ap.pcolor_latlon(u[t,k])
plt.subplot(212)
ap.pcolor_latlon(v[t,k])

# ----------------------------------------------------------------------
# Moisture fluxes

uq = u * q
vq = v * q

uq_int = dat.int_pres(uq, pdim=-3)
vq_int = dat.int_pres(vq, pdim=-3)

t, k = 0, 4
plt.figure(figsize=(7,8))
plt.subplot(211)
ap.pcolor_latlon(uq[t,k])
plt.subplot(212)
ap.pcolor_latlon(vq[t,k])

plt.figure(figsize=(7,8))
plt.subplot(211)
ap.pcolor_latlon(uq_int[t])
plt.subplot(212)
ap.pcolor_latlon(vq_int[t])

# ----------------------------------------------------------------------
# Moisture flux convergence

# DataArray
mfc = - divergence_spherical_2d(uq_int, vq_int)

# ndarray
mfc2 = - divergence_spherical_2d(uq_int.values, vq_int.values, lat, lon)

# components
mfc3, mfc_x, mfc_y = divergence_spherical_2d(uq_int, vq_int, return_comp=True)
mfc3, mfc_x, mfc_y = -mfc3, -mfc_x, -mfc_y

# Convert from (kg/m^2)/s to mm/day
scale = 60 * 60 * 24
mfc *= scale
mfc2 *= scale
mfc3 *= scale
mfc_x *= scale
mfc_y *= scale

cmax = 12
plt.figure(figsize=(7,8))
plt.subplot(211)
ap.pcolor_latlon(mfc)
plt.clim(-cmax, cmax)
plt.subplot(212)
ap.pcolor_latlon(mfc2, lat, lon)
plt.clim(-cmax, cmax)

plt.figure(figsize=(7,8))
plt.subplot(211)
ap.pcolor_latlon(mfc_x)
plt.clim(-cmax, cmax)
plt.subplot(212)
ap.pcolor_latlon(mfc_y)
plt.clim(-cmax, cmax)
