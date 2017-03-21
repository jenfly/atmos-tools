# Standard scientific modules:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import xarray as xray
from datetime import datetime
import statsmodels.api as sm
import scipy.signal as signal
import collections
from __future__ import division

# My modules:
import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
import atmos.variables as av
from atmos.utils import print_if
from atmos.constants import const as constants
from atmos.data import get_coord

# ----------------------------------------------------------------------
# Gradients

n = 100
x = np.linspace(0, 2*np.pi, n)
x[5] = np.nan
x[60:70] = np.nan
y = np.sin(x)
dydx = np.gradient(y, np.gradient(x))

plt.figure()
plt.plot(x, y, 'b')
plt.plot(x, dydx, 'r')
plt.plot(x, np.cos(x), 'k--')

# ----------------------------------------------------------------------
# Linear regression

n = 100
x = np.linspace(0,10,n)
y = 2*x - 5 + 10*np.random.random(n)
plt.figure()
plt.scatter(x, y)
x_in = sm.add_constant(x)
fit = sm.OLS(y, x_in).fit()
ypred = fit.predict(x_in)
plt.plot(x, ypred, 'r')
print fit.summary()

# ----------------------------------------------------------------------
# Multiple linear regression

nobs = 100
x = np.random.random((nobs, 2))
x = sm.add_constant(x)
beta = [1, .1, .5]
e = np.random.random(nobs)
y = np.dot(x, beta) + e
results = sm.OLS(y, x).fit()
print results.summary()
# ----------------------------------------------------------------------
# Read some data from OpenDAP url

url = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA_MONTHLY/'
    'MAIMCPASM.5.2.0/1979/MERRA100.prod.assim.instM_3d_asm_Cp.197901.hdf')

ds = xray.open_dataset(url)
T = ds['T']
ps = ds['PS']
q = ds['QV']
plev = get_coord(T, 'plev')
lat = get_coord(T, 'lat')
lon = get_coord(T, 'lon')

# t, k = 0, 6
# t, k = 0, 22
t, k = 0, 14
pstr = '%d hPa' % (plev[k]/100)

# ----------------------------------------------------------------------
