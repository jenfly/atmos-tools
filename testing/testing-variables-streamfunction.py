import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
from atmos.constants import const as constants
from atmos.data import get_coord
from atmos.variables import streamfunction


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
# Streamfunction - top-down, zonal mean

# DataArray
psi = streamfunction(v)
psibar = psi.mean(axis=-1)

# ndarray
psi2 = streamfunction(v.values, lat, plev * 100)
psibar2 = np.nanmean(psi2, axis=-1)

topobar = topo.mean(axis=-1)

cint = 50
plt.figure(figsize=(7,8))
plt.subplot(211)
ap.contour_latpres(psibar, clev=cint, topo=topobar)
plt.subplot(212)
ap.contour_latpres(psibar2, lat, plev, clev=cint, topo=topobar)

# ----------------------------------------------------------------------
# Sector mean, top-down

lon1, lon2 = 60, 100
cint=10

vbar = dat.subset(v, 'lon', lon1, lon2).mean(axis=-1)

psi = streamfunction(v)
psibar = dat.subset(psi, 'lon', lon1, lon2).mean(axis=-1)
psibar = psibar * (lon2-lon1)/360

psibar2 = streamfunction(vbar)
psibar2 = psibar2 * (lon2-lon1)/360

topobar = dat.subset(topo, 'lon', lon1, lon2).mean(axis=-1)

plt.figure(figsize=(7,8))
plt.subplot(211)
ap.contour_latpres(psibar, clev=cint, topo=topobar)
plt.subplot(212)
ap.contour_latpres(psibar2, clev=cint, topo=topobar)

# Note: streamfunction() doesn't work properly when you take the
# zonal/sector mean of v before computing the streamfunction.

# ----------------------------------------------------------------------
