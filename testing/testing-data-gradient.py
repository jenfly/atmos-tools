import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import atmos as atm
import precipdat
import merra

# ----------------------------------------------------------------------

datadir = atm.homedir() + 'datastore/merra/daily/'
year = 2014
subset = '_40E-120E_90S-90N'

def get_var(datadir, varnm, subset, year):
    filenm = '%smerra_%s%s_%d.nc' % (datadir, varnm, subset, year)
    with xray.open_dataset(filenm) as ds:
        var = ds[varnm].load()
    return var

uq_int = get_var(datadir, 'UFLXQV', subset, year)
vq_int = get_var(datadir, 'VFLXQV', subset, year)

mfc = atm.moisture_flux_conv(uq_int, vq_int, already_int=True)
mfcbar = mfc.mean(dim='YDim').mean(dim='XDim')

# Test atm.gradient
a = atm.constants.radius_earth.values
latdim, londim = 1, 2
lat = atm.get_coord(uq_int, 'lat')
latrad = np.radians(lat)
latrad[abs(lat) > 89] = np.nan
coslat = xray.DataArray(np.cos(latrad), coords={'YDim' : lat})
lon = atm.get_coord(uq_int, 'lon')
lonrad = np.radians(lon)

mfc_x = atm.gradient(uq_int, lonrad, londim) / (a*coslat)
mfc_y = atm.gradient(vq_int * coslat, latrad, latdim) / (a*coslat)
mfc_test  = mfc_x + mfc_y
mfc_test = - atm.precip_convert(mfc_test, 'kg/m2/s', 'mm/day')
mfc_test_bar = mfc_test.mean(dim='YDim').mean(dim='XDim')

diff = mfc_test - mfc
print(diff.max())
print(diff.min())

plt.plot(mfcbar)
plt.plot(mfc_test_bar)
print(mfc_test_bar - mfcbar)

# ----------------------------------------------------------------------
# Vertical gradient du/dp

lon1, lon2 = 40, 120
pmin, pmax = 100, 300
subset_dict = {'XDim' : (lon1, lon2), 'Height' : (pmin, pmax)}

urls = merra.merra_urls([year])
month, day = 7, 15
url = urls['%d%02d%02d' % (year, month, day)]
with xray.open_dataset(url) as ds:
    u = atm.subset(ds['U'], subset_dict, copy=False)
    u = u.mean(dim='TIME')

pres = u['Height']
pres = atm.pres_convert(pres, pres.attrs['units'], 'Pa')
dp = np.gradient(pres)

# Calc 1
dims = u.shape
dudp = np.nan * u
for i in range(dims[1]):
    for j in range(dims[2]):
        dudp.values[:, i, j] = np.gradient(u[:, i, j], dp)

# Test atm.gradient
dudp_test = atm.gradient(u, pres, axis=0)
diff = dudp_test - dudp
print(diff.max())
print(diff.min())
