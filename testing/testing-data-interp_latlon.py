"""Testing  interp_latlon"""

import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
from atmos.data import get_coord, latlon_equal, lon_convention, set_lon

# ----------------------------------------------------------------------
# NCEP2
# ----------------------------------------------------------------------

filename = 'data/more/ncep2_climatology_monthly.nc'
ds = dat.ncload(filename)
mon = ds['mon'].values
u = ds['u']
v = ds['v']
T = ds['T']
ps = ds['ps']
lat = get_coord(ps, 'lat')
lon = get_coord(ps, 'lon')
plev = get_coord(u, 'plev')

topo = dat.get_ps_clim(lat, lon) / 100
topo.units = 'hPa'

# ----------------------------------------------------------------------
# MERRA
# ----------------------------------------------------------------------

url = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA_MONTHLY/'
    'MAIMCPASM.5.2.0/1979/MERRA100.prod.assim.instM_3d_asm_Cp.197901.hdf')

ds_m = xray.open_dataset(url)
T_m = ds_m['T']
ps_m = ds_m['PS']
q_m = ds_m['QV']
plev_m = get_coord(T_m, 'plev')
lat_m = get_coord(T_m, 'lat')
lon_m = get_coord(T_m, 'lon')

# ----------------------------------------------------------------------
# latlon_equal, lon_convention, set_lon

print(latlon_equal(ps, T))
print(latlon_equal(T, T_m))

print(lon_convention(lon))
print(lon_convention(lon_m))

ps_m2 = set_lon(ps_m, lonmax=360)

t = 0
x1, y1 = np.meshgrid(lon_m, lat_m)
x2, y2 = np.meshgrid(ps_m2.XDim, ps_m2.YDim)
cmap='jet'
ncontours = 30

plt.figure(figsize=(7,8))
plt.subplot(211)
plt.contourf(x1, y1, ps_m[t].values, ncontours, cmap=cmap)
plt.colorbar()
plt.subplot(212)
plt.contourf(x2, y2, ps_m2[t].values, ncontours, cmap=cmap)
plt.colorbar()

plt.figure(figsize=(7,8))
plt.subplot(211)
ap.pcolor_latlon(ps_m[t], cmap=cmap)
plt.subplot(212)
ap.pcolor_latlon(ps_m2[t], cmap=cmap)
