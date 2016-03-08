import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos as atm

# ----------------------------------------------------------------------
# Monthly climatology

yearstr = '1979-2015'
varnms = ['U', 'V']
datadir = atm.homedir() + 'datastore/merra/monthly/'
filestr = datadir + 'merra_%s_%s.nc'
files = {nm : filestr % (nm, yearstr) for nm in varnms}

data = xray.Dataset()
for nm in varnms:
    with xray.open_dataset(files[nm]) as ds:
        data[nm] = ds[nm].load()

lat = atm.get_coord(data, 'lat')
lon = atm.get_coord(data, 'lon')
psfile = atm.homedir() + 'dynamics/python/atmos-tools/data/topo/ncep2_ps.nc'
ps = atm.get_ps_clim(lat, lon, psfile)
ps = ps / 100

figsize = (7, 9)
omitzero = False

for ssn in ['ANN', 'DJF', 'JJA', 'MAR']:
    for lonlims in [(0, 360), (60, 100)]:
        lon1, lon2 = lonlims
        lonstr = atm.latlon_str(lon1, lon2, 'lon')
        suptitle = ssn + ' ' + lonstr
        months = atm.season_months(ssn)
        v = data['V'].sel(month=months)
        if (lon2 - lon1) < 360:
            v = atm.subset(v, {'lon' : (lon1, lon2)})
            sector_scale = (lon2 - lon1) / 360.0
            psbar = atm.dim_mean(ps, 'lon', lon1, lon2)
            clev = 10
        else:
            sector_scale = None
            psbar = atm.dim_mean(ps, 'lon')
            clev = 20
        vssn = v.mean(dim='month')
        vssn_bar = atm.dim_mean(vssn, 'lon')
        psi1 = atm.streamfunction(vssn, sector_scale=sector_scale)
        psi1 = atm.dim_mean(psi1, 'lon')
        psi2 = atm.streamfunction(vssn_bar, sector_scale=sector_scale)
        plt.figure(figsize=figsize)
        plt.suptitle(suptitle)
        plt.subplot(2, 1, 1)
        atm.contour_latpres(psi1, clev=clev, omitzero=omitzero, topo=psbar)
        plt.title('v -> $\psi$ -> [$\psi$]')
        plt.xlabel('')
        plt.subplot(2, 1, 2)
        atm.contour_latpres(psi2, clev=clev, omitzero=omitzero, topo=psbar)
        plt.title('[v] -> [$\psi$]')

# ----------------------------------------------------------------------
# Daily data

datadir = atm.homedir() + 'datastore/merra/daily/'
filestr = datadir + 'merra_V_sector_%s_%d.nc'
year = 1980
#lon1, lon2 = 0, 360
lon1, lon2 = 60, 100
lonstr = atm.latlon_str(lon1, lon2, 'lon')

filenm = filestr % (lonstr, year)
with xray.open_dataset(filenm) as ds:
    v = ds['V'].load()

if (lon2 - lon1) < 360:
    sector_scale = (lon2 - lon1) / 360.
else:
    sector_scale = None

clev = 10
#clev = 20
#ssn = 'ANN'
#days = atm.season_days(ssn)
days = range(170, 176)
vbar = v.sel(day=days).mean(dim='day')
psi = atm.streamfunction(vbar, sector_scale=sector_scale)
plt.figure()
atm.contour_latpres(psi, clev=clev, omitzero=False)




# ======================================================================
#
# # ----------------------------------------------------------------------
# # Read data
# url = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA_MONTHLY/'
#     'MAIMCPASM.5.2.0/1979/MERRA100.prod.assim.instM_3d_asm_Cp.197907.hdf')
#
# ds = xray.open_dataset(url)
# #T = ds['T']
# #ps = ds['PS']
# #u = ds['U']
# v = ds['V']
# #q = ds['QV']
# lat = get_coord(v, 'lat')
# lon = get_coord(v, 'lon')
# plev = get_coord(v, 'plev')
#
# topo = dat.get_ps_clim(lat, lon) / 100
# topo.units = 'hPa'
#
# # ----------------------------------------------------------------------
# # Streamfunction - top-down, zonal mean
#
# # DataArray
# psi = streamfunction(v)
# psibar = psi.mean(axis=-1)
#
# # ndarray
# psi2 = streamfunction(v.values, lat, plev * 100)
# psibar2 = np.nanmean(psi2, axis=-1)
#
# topobar = topo.mean(axis=-1)
#
# cint = 50
# plt.figure(figsize=(7,8))
# plt.subplot(211)
# ap.contour_latpres(psibar, clev=cint, topo=topobar)
# plt.subplot(212)
# ap.contour_latpres(psibar2, lat, plev, clev=cint, topo=topobar)
#
# # ----------------------------------------------------------------------
# # Sector mean, top-down
#
# lon1, lon2 = 60, 100
# cint=10
#
# vbar = dat.subset(v, 'lon', lon1, lon2).mean(axis=-1)
#
# psibar = dat.subset(psi, 'lon', lon1, lon2).mean(axis=-1)
# psibar = psibar * (lon2-lon1)/360
#
# psibar2 = streamfunction(vbar)
# psibar2 = psibar2 * (lon2-lon1)/360
#
# topobar = dat.subset(topo, 'lon', lon1, lon2).mean(axis=-1)
#
# plt.figure(figsize=(7,8))
# plt.subplot(211)
# ap.contour_latpres(psibar, clev=cint, topo=topobar)
# plt.subplot(212)
# ap.contour_latpres(psibar2, clev=cint, topo=topobar)
#
# # Note: streamfunction() doesn't work properly when you take the
# # zonal/sector mean of v before computing the streamfunction.
#
# # ----------------------------------------------------------------------
# # Top-down and bottom-up
#
# psibar = psi.mean(axis=-1)
# psi2 = streamfunction(v, topdown=False)
# psibar2 = psi2.mean(axis=-1)
#
# topobar = topo.mean(axis=-1)
#
# cint = 50
# plt.figure(figsize=(7,8))
# plt.subplot(211)
# ap.contour_latpres(psibar, clev=cint, topo=topobar)
# plt.subplot(212)
# ap.contour_latpres(psibar2, lat, plev, clev=cint, topo=topobar)
