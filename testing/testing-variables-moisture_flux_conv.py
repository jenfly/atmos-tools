import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
from atmos.constants import const as constants
from atmos.data import get_coord
from atmos.variables import moisture_flux_conv


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
# Moisture flux convergence

uq = u * q
vq = v * q

# DataArray
mfc = moisture_flux_conv(uq,vq)

# ndarray
mfc2 = moisture_flux_conv(uq.values, vq.values, lat, lon, plev*100)

# components
mfc3, mfc_x, mfc_y, uq_int, vq_int = moisture_flux_conv(
    uq, vq, return_comp=True)

# pmin, pmax
mfc_sub = moisture_flux_conv(uq, vq, pmin=600e2, pmax=900e2)

cmax = 12
plt.figure(figsize=(7,10))
plt.subplot(311)
ap.pcolor_latlon(mfc)
plt.clim(-cmax, cmax)
plt.subplot(312)
ap.pcolor_latlon(mfc2, lat, lon)
plt.clim(-cmax, cmax)
plt.subplot(313)
ap.pcolor_latlon(mfc_sub)
plt.clim(-cmax, cmax)

plt.figure(figsize=(7,8))
plt.subplot(211)
ap.pcolor_latlon(mfc_x)
plt.clim(-cmax, cmax)
plt.subplot(212)
ap.pcolor_latlon(mfc_y)
plt.clim(-cmax, cmax)
