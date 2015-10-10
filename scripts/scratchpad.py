# Standard scientific modules:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import xray
from datetime import datetime
import statsmodels.api as sm

# My modules:
import atmos.utils as utils
import atmos.plots as ap
import atmos.data as dat
import atmos.variables as av
from atmos.utils import print_if
from atmos.constants import const as constants
from atmos.data import get_coord



# ----------------------------------------------------------------------
# Harmonic analysis

def harmonic_decomp(y, dt=1.0):
    n = len(y)
    T = float(n * dt)
    nhalf = n // 2
    ts = np.arange(1, n+1)
    omega1 = 2 * np.pi / n

    harmonics = {}
    harmonics[0] = np.mean(y)
    freqs = np.zeros(nhalf)
    Ck = np.zeros(nhalf)
    for k in range(1, nhalf + 1):
        omega = k * omega1
        A = (2.0/n) * np.sum(y * np.cos(omega * ts))
        B = (2.0/n) * np.sum(y * np.sin(omega * ts))
        freqs[k-1] = k / T
        Ck[k-1] = np.sqrt(A**2 + B**2)
        harmonics[k] = A*np.cos(omega*ts) + B*np.sin(omega*ts)

    return freqs, Ck, harmonics

# -----------
N = 365
dt = 1.0 / N
x = np.arange(1, N+1)
omega1 = 2 * np.pi / N
y = 3.1 + 2.5 * np.sin(omega1 * x) + np.cos(2 * omega1 * x)

freqs, Ck, harmonics = harmonic_decomp(y, dt)
plt.figure(figsize=(7,8))
plt.subplot(211)
plt.plot(x,y)
plt.plot(x, harmonics[1])
plt.plot(x, harmonics[2])
plt.plot(x, harmonics[0] + harmonics[1] + harmonics[2], '--')
plt.subplot(212)
plt.plot(freqs[:10], Ck[:10])



# -------------------
datadir = '/home/jennifer/datastore/cmap/'
with xray.open_dataset(datadir + 'cmap.precip.pentad.mean.nc') as ds:
    ds.load()
cmap = ds['precip'].sel(lat=11.25, lon=91.25)

npentad = 73 # pentads/year
dt = 5.0/365
nyears = 3
precip = cmap[:nyears*npentad]
freqs, Ck, harmonics = harmonic_decomp(precip.values, dt)
ntrunc = 12
ytrunc = harmonics[0]
for k in range(1,nharm+1):
    ytrunc += harmonics[k]
ypred = np.zeros(len(precip))
for k in harmonics:
    ypred += harmonics[k]

t = precip.time
plt.figure(figsize=(7,8))
plt.subplot(211)
plt.plot(t,precip)
for k in range(1, trunc+1):
    plt.plot(t, harmonics[k])
plt.plot(t, ytrunc, '--')
plt.plot(t, ypred, '--')
plt.subplot(212)
plt.plot(freqs[:10], Ck[:10])


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
y = 2*x - 5 + 2*np.random.random(n)
plt.figure()
plt.scatter(x, y)
x_in = sm.add_constant(x)
fit = sm.OLS(y, x_in).fit()

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
