import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
from atmos.constants import const as constants
from atmos.data import get_coord
from atmos.variables import rel_vorticity


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
# Relative vorticity

# DataArray
vort = rel_vorticity(u, v)

# ndarray
vort2 = rel_vorticity(u, v, lat, lon)


t, k = 0, 22
plt.figure(figsize=(7,8))
plt.subplot(211)
ap.pcolor_latlon(vort[t,k])
plt.subplot(212)
ap.pcolor_latlon(vort2[t,k], lat, lon)
