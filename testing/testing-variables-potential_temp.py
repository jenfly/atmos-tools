# Standard scientific modules:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import xray
from datetime import datetime

# My modules:
import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
import atmos.variables as av
from atmos.utils import print_if
from atmos.constants import const as constants

# ----------------------------------------------------------------------
# Read some data from OpenDAP url

url = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA_MONTHLY/'
    'MAIMCPASM.5.2.0/1979/MERRA100.prod.assim.instM_3d_asm_Cp.197901.hdf')

ds = xray.open_dataset(url)
T = ds['T']
ps = ds['PS']
q = ds['QV']
plev = dat.get_plev(T, units='Pa')
lat, lon = dat.get_lat(ps), dat.get_lon(ps)
p0 = 1e5

print('Calculating potential temperature')
theta = av.potential_temp(T, plev, p0)
print('Calculating equivalent potential temperature')
theta_e = av.equiv_potential_temp(T, plev, q, p0)

# t, k = 0, 6
# t, k = 0, 22
t, k = 0, 14
pstr = '%d hPa' % (plev[k]/100)
plt.figure(figsize=(7,10))
plt.subplot(311)
ap.pcolor_latlon(T[t,k])
plt.title('Temperature ' + pstr)
plt.subplot(312)
ap.pcolor_latlon(theta[t,k])
plt.title('Potential Temperature ' + pstr)
plt.subplot(313)
ap.pcolor_latlon(theta_e[t,k])
plt.title('Equiv Potential Temperature ' + pstr)

ps.dims
plt.figure()
ap.pcolor_latlon(ps,cmap='hot')
