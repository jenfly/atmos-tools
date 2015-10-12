import xray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import atmos as atm
from atmos import Fourier, fourier_from_scratch

def test_fourier(spec, ntrunc):
    harm = {}
    nf = len(spec.f_k)
    for k in range(nf):
        harm[k] = spec1.harmonic(k)
    ypred = spec.smooth(nf)
    ysmooth = spec.smooth(ntrunc)
    spec.harm, spec.ypred, spec.ysmooth = harm, ypred, ysmooth
    return spec

def plot_fourier(spec, nharm):
    plt.figure(figsize=(7,8))
    plt.subplot(211)
    plt.plot(spec.t, spec.tseries)
    for k in range(1, nharm+1):
        plt.plot(spec.t, spec.harm[k])
    plt.plot(spec.t, spec.ysmooth, 'r')
    plt.plot(spec.t, spec.ypred, '--')
    plt.subplot(212)
    plt.plot(spec.f_k, spec.ps_k)

class Struct(object):
    pass

def fromscratch(y, dt, ntrunc, t):
    sp = Struct()
    sp.t = t
    sp.tseries = y
    output = fourier_from_scratch(y, dt, ntrunc)
    sp.f_k, sp.C_k, sp.ps_k, sp.harm, sp.ypred, sp.ysmooth = output
    return sp

# ----------------------------------------------------------------------
N = 365
dt = 1.0 / N
x = np.arange(1, N+1)
omega1 = 2 * np.pi / N
y = 3.1 + 2.5 * np.sin(omega1 * x) + np.cos(2 * omega1 * x)
ntrunc = 1

spec1 = Fourier(y, dt)
spec1 = test_fourier(spec1, ntrunc)
spec2 = fromscratch(y, dt, ntrunc, spec1.t)
plot_fourier(spec1, ntrunc)
plot_fourier(spec2, ntrunc)

# ----------------------------------------------------------------------
df = pd.read_csv('data/SOI_index.csv',header=4, index_col=0)
soi = df.stack()
plt.figure()
soi.plot()
y = soi.values
dt = 1.0/12
t = 1876 + np.arange(1, len(y)+1)*dt
ntrunc = 40
nharm = 3 # Number of harmonics to plot

spec1 = Fourier(y, dt, t)
spec1 = test_fourier(spec1, ntrunc)
spec2 = fromscratch(y, dt, ntrunc, spec1.t)
plot_fourier(spec1, nharm)
plot_fourier(spec2, nharm)

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
nharm = 3

spec1 = Fourier(y, dt)
spec1 = test_fourier(spec1, ntrunc)
spec2 = fromscratch(y, dt, ntrunc, spec1.t)
plot_fourier(spec1, nharm)
plot_fourier(spec2, nharm)
