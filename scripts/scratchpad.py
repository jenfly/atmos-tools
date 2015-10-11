# Standard scientific modules:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import xray
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
# Harmonic analysis

# ----------------------------------------------------------------------

class Fourier:
    def __init__(self, y, dt=1.0, t=None, normalize_ps=True, time_units=None,
                 data_name=None):

        n = len(y)
        if t is None:
            t = dt * np.arange(n)
        self.t = t
        self.tseries = y
        self.f_k = np.fft.rfftfreq(n, dt)
        self.tau_k = np.concatenate(([np.nan], 1/self.f_k[1:]))
        self.attrs = {'data_name' : data_name, 'time_units' : time_units,
                      'dt' : dt, 'normalize_ps' : normalize_ps}

        # Fourier coefficients
        self.C_k = np.fft.rfft(y)

        # Power spectral density
        ps_k = 2 * np.abs(self.C_k / n)**2
        if normalize_ps:
            ps_k = n * ps_k / np.sum((y - np.mean(y))**2)
        self.ps_k = ps_k


    def __repr__(self):

        def var_str(name, x):
            width = 10
            return '%s [%d] : %f, %f, ..., %f\n' % (name.ljust(width), len(x),
                                                    x[0], x[1], x[-1])

        s = 'Attributes\n' + str(self.attrs) + '\n\nData\n'
        s = s + (var_str('t', self.t) + var_str('tseries', self.tseries) +
                 var_str('f_k', self.f_k) + var_str('tau_k', self.tau_k) +
                 var_str('C_k', self.C_k) + var_str('ps_k', self.ps_k))

        return s

    def smooth(self, kmax):
        return np.fft.irfft(self.C_k[:kmax+1], len(self.tseries))

    def harmonic(self, k):
        if k == 0:
            y = self.smooth(k)
        else:
            y = self.smooth(k) - self.smooth(k - 1)
        return y


# ----------------------------------------------------------------------


def fft_spectrum(y, dt=1.0, normalize=False):
    n = len(y)
    Ck = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, dt)
    ypred = np.fft.irfft(Ck, n)

    # Power spectrum
    ps = np.abs(Ck / n)**2

    # Account for pos/neg frequencies (exclude 0 and Nyquist frequency because
    # they only appear once in the full spectrum)
    ps[1:-1] = 2 * ps[1:-1]

    if normalize:
        ps = n * ps / np.sum((y - np.mean(y))**2)

    return freqs, ps, Ck, ypred

def fft_smooth(y, kmax):
    Ck = np.fft.rfft(y)
    ysmooth = np.fft.irfft(Ck[:kmax+1], len(y))
    return ysmooth

def fourier_from_scratch(y, dt=1.0, ntrunc=None):
    """ Wilks Atm Stats eqs..."""

    # Make sure we're working with an ndarray and not a DataArray
    if isinstance(y, xray.DataArray):
        y = y.values

    n = len(y)
    nhalf = n // 2
    t = np.arange(1, n+1)
    omega1 = 2 * np.pi / n

    # Fourier transform and harmonics
    Ak = np.zeros(nhalf + 1, dtype=float)
    Bk = np.zeros(nhalf + 1, dtype=float)
    Ak[0] = np.mean(y)
    Bk[0] = 0.0
    harmonics = {0 : np.mean(y)}
    for k in range(1, nhalf + 1):
        omega = k * omega1
        Ak[k] = (2.0/n) * np.sum(y * np.cos(omega*t))
        Bk[k] = (2.0/n) * np.sum(y * np.sin(omega*t))
        harmonics[k] = Ak[k] * np.cos(omega*t) + Bk[k] * np.sin(omega*t)

    # Fourier coefficients
    Ck = Ak + 1j * Bk

    # Normalized power spectrum
    ps = nhalf * np.abs(Ck)**2 / np.sum((y - np.mean(y))**2)

    # Frequencies
    freqs = np.arange(n//2 + 1) / float(n * dt)

    # Predicted y and smoothed truncated predicted y
    ypred = np.zeros(n, dtype=float)
    ytrunc = np.zeros(n, dtype=float)
    for k in harmonics:
        ypred += harmonics[k]
        if ntrunc is not None and k <= ntrunc:
            ytrunc += harmonics[k]

    return freqs, harmonics, Ck, ps, ypred, ytrunc

# ----------------------------------------------------------------------
N = 365
dt = 1.0 / N
x = np.arange(1, N+1)
omega1 = 2 * np.pi / N
y = 3.1 + 2.5 * np.sin(omega1 * x) + np.cos(2 * omega1 * x)

ntrunc = 1
freqs, harmonics, Ck, ps, ypred, ytrunc = fourier_from_scratch(y, dt, ntrunc)
plt.figure(figsize=(7,8))
plt.subplot(211)
plt.plot(x,y)
plt.plot(x, harmonics[1])
plt.plot(x, harmonics[2])
plt.plot(x, ytrunc)
plt.plot(x, ypred, 'r--')
plt.subplot(212)
plt.plot(freqs, ps)
plt.xlim(0, 5)

# Compare with FFT
freqs2, ps2, Ck2, ypred2 = fft_spectrum(y, dt, normalize=True)
ytrunc2 = fft_smooth(y, ntrunc)
plt.figure(figsize=(7,8))
plt.subplot(211)
plt.plot(x,y)
plt.plot(x,ytrunc2)
plt.plot(x, ypred2, '--')
plt.subplot(212)
plt.plot(freqs2, ps2)
plt.xlim(0, 5)

# ----------------------------------------------------------------------

df = pd.read_csv('data/SOI_index.csv',header=4, index_col=0)
soi = df.stack()

plt.figure()
soi.plot()

y = soi.values
dt = 1.0/12
t = 1876 + np.arange(1, len(y)+1)*dt


ntrunc = 40
freqs, harmonics, Ck, ps, ypred, ytrunc = fourier_from_scratch(y, dt, ntrunc)
plt.figure(figsize=(14,8))
plt.subplot(211)
plt.plot(t,y)
plt.plot(t, ytrunc)
plt.plot(t, ypred, 'r--')
plt.subplot(212)
plt.plot(freqs, ps)

# Compare with FFT
freqs2, ps2, Ck2, ypred2 = fft_spectrum(y, dt, normalize=True)
ytrunc2 = fft_smooth(y, ntrunc)
plt.figure(figsize=(14,8))
plt.subplot(211)
plt.plot(t,y)
plt.plot(t,ytrunc2)
plt.plot(t, ypred2, '--')
plt.subplot(212)
plt.plot(freqs2, ps2)

# ----------------------------------------------------------------------
datadir = '/home/jennifer/datastore/cmap/'
with xray.open_dataset(datadir + 'cmap.precip.pentad.mean.nc') as ds:
    ds.load()
cmap = ds['precip'].sel(lat=11.25, lon=91.25)

npentad = 73 # pentads/year
dt = 5.0/365
nyears = 1
precip = cmap[:nyears*npentad]
y = precip.values
ntrunc = 12
freqs, harmonics, Ck, Rk_sq, ypred, ytrunc = fourier_from_scratch(y, dt, ntrunc)

t = precip.time
plt.figure(figsize=(7,8))
plt.subplot(211)
plt.plot(t,y)
for k in range(1, ntrunc+1):
    plt.plot(t, harmonics[k])
plt.plot(t, ytrunc, '--')
plt.plot(t, ypred, 'r--')
plt.subplot(212)
plt.plot(freqs, np.abs(Ck)**2)

# Compare with FFT
Ck2 = np.fft.rfft(y)
freqs2 = np.fft.rfftfreq(len(y), dt)
ypred2 = np.fft.irfft(Ck2, len(y))
ytrunc2 = np.fft.irfft(Ck2[:ntrunc+1], len(y))
plt.figure(figsize=(7,8))
plt.subplot(211)
plt.plot(t,y)
plt.plot(t,ytrunc2, '--')
plt.plot(t, ypred2, 'r--')
plt.subplot(212)
plt.plot(freqs2, np.abs(Ck2)**2)

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
