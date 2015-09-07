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
from atmos.utils import print_if

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
ps = ds['PS']
ps.dims
plt.figure()
ap.pcolor_latlon(ps,cmap='hot')


# ----------------------------------------------------------------------
# Opening multiple OpenDAP files into xray dataset

url_dir = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA/'
    'MAI3CPASM.5.2.0/1979/01/')
start = 'MERRA100.prod.assim.inst3_3d_asm_Cp.197901'
#end = '.hdf?U,V,XDim,YDim,Height,TIME'
end = '.hdf'

paths = ['%s%s%02d%s' % (url_dir, start, i, end) for i in range(1,3)]

var = 'U'
concat_dim = 'TIME'
u = dat.load_concat(paths, var, concat_dim=concat_dim, verbose=True)

# v = ds['V']
# T = ds['T']
# ps = ds['PS']
# q = ds['QV']
# hgt = ds['H']
# omega = ds['OMEGA']

# uu = u * u
# uv = u * v

ds_out = xray.Dataset()
ds_out['u'] = u.mean(axis=0)
# ds_out['v'] = v.mean(axis=0)
# ds_out['uu'] = uu.mean(axis=0)
# ds_out['uv'] = uv.mean(axis=0)

outfile = 'data/more/merra_uv_197901.nc'
ds_out.to_netcdf(outfile,mode='w')
