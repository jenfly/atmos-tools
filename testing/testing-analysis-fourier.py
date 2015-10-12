import xray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import atmos as atm
from atmos import Fourier, fourier_from_scratch

#datadir = '/home/jennifer/datastore/cmap/'
datadir = '/home/jwalker/datastore/cmap/'
cmap_file = datadir + 'cmap.precip.pentad.mean.nc'

def test_fourier(spec, ntrunc):
    harm = {}
    nf = len(spec.f_k)
    for k in range(nf):
        harm[k] = spec1.harmonic(k)
    ypred = spec.smooth(nf)
    ysmooth = spec.smooth(ntrunc)
    spec.harm, spec.ypred, spec.ysmooth = harm, ypred, ysmooth
    spec.Rsq = spec.Rsquared()
    return spec

def plot_fourier(spec, nharm, ntrunc):
    plt.figure(figsize=(12,10))
    plt.subplot(311)
    plt.plot(spec.t, spec.tseries)
    for k in range(1, nharm+1):
        plt.plot(spec.t, spec.harm[k])
    plt.plot(spec.t, spec.ysmooth, 'r')
    plt.plot(spec.t, spec.ypred, '--')
    plt.subplot(312)
    plt.plot(spec.f_k, spec.ps_k)
    plt.subplot(313)
    plt.plot(spec.f_k[:ntrunc+1], spec.Rsq[:ntrunc+1])
    print(np.sum(spec.Rsq))
    print(np.sum(spec.Rsq[:ntrunc+1]))
    

class Struct(object):
    pass

def fromscratch(y, dt, ntrunc, t):
    sp = Struct()
    sp.t = t
    sp.tseries = y
    output = fourier_from_scratch(y, dt, ntrunc)
    sp.f_k, sp.C_k, sp.ps_k, sp.harm, sp.ypred, sp.ysmooth = output
    sp.Rsq = sp.ps_k * len(y) / np.sum((y - np.mean(y))**2)
    sp.Rsq[0] = 0.0
    return sp

# ----------------------------------------------------------------------
N = 365
dt = 1.0 / N
x = np.arange(1, N+1)
omega1 = 2 * np.pi / N
y = 3.1 + 2.5 * np.sin(omega1 * x) + np.cos(2 * omega1 * x)
ntrunc = 5

spec1 = Fourier(y, dt)
spec1 = test_fourier(spec1, ntrunc)
spec2 = fromscratch(y, dt, ntrunc, spec1.t)
plot_fourier(spec1, ntrunc, ntrunc)
plot_fourier(spec2, ntrunc, ntrunc)

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
plot_fourier(spec1, nharm, ntrunc)
plot_fourier(spec2, nharm, ntrunc)

# ----------------------------------------------------------------------

with xray.open_dataset(cmap_file) as ds:
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
plot_fourier(spec1, nharm, ntrunc)
plot_fourier(spec2, nharm, ntrunc)
