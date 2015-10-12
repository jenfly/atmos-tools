"""
Functions for atmospheric data analysis.

- Spectral analysis
- Timeseries
- Linear regression
"""

from __future__ import division
import numpy as np
import collections
import xray

#import atmos.utils as utils
#import atmos.xrhelper as xr

# ======================================================================
# SPECTRAL ANALYSIS
# ======================================================================

class Fourier:
    def __init__(self, y, dt=1.0, t=None, normalize_ps=False, time_units=None,
                 data_name=None):

        # Make sure we're working with an ndarray and not a DataArray
        if isinstance(y, xray.DataArray):
            y = y.values

        self.attrs = {'data_name' : data_name, 'time_units' : time_units,
                      'dt' : dt, 'normalize_ps' : normalize_ps}
        n = len(y)
        if t is None:
            t = dt * np.arange(n)
        self.t = t
        self.tseries = y

        # Fourier frequencies and coefficients
        self.f_k = np.fft.rfftfreq(n, dt)
        self.C_k = np.fft.rfft(y)

        # Periods corresponding to Fourier frequencies
        self.tau_k = np.concatenate(([np.nan], 1/self.f_k[1:]))

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
def fourier_from_scratch(y, dt=1.0, ntrunc=None):
    """ Wilks Atm Stats eqs..."""

    n = len(y)
    nhalf = n // 2
    t = np.arange(1, n+1)
    omega1 = 2 * np.pi / n

    # Fourier transform and harmonics
    Ak = np.zeros(nhalf + 1, dtype=float)
    Bk = np.zeros(nhalf + 1, dtype=float)
    #Ak[0] = np.mean(y)
    Ak[0] = np.sum(y) / 2.0
    Bk[0] = 0.0
    harmonics = {0 : np.mean(y)}
    for k in range(1, nhalf + 1):
        omega = k * omega1
        # Ak[k] = (2.0/n) * np.sum(y * np.cos(omega*t))
        # Bk[k] = (2.0/n) * np.sum(y * np.sin(omega*t))
        # harmonics[k] = Ak[k] * np.cos(omega*t) + Bk[k] * np.sin(omega*t)
        Ak[k] = np.sum(y * np.cos(omega*t))
        Bk[k] = np.sum(y * np.sin(omega*t))
        harmonics[k] = ((2.0/n) * Ak[k] * np.cos(omega*t) +
                        (2.0/n) * Bk[k] * np.sin(omega*t))

    # Fourier coefficients
    C_k = Ak + 1j * Bk

    # Frequencies
    f_k = np.arange(n//2 + 1) / float(n * dt)

    # Power spectral density
    ps_k = 2 * np.abs(C_k / n)**2

    # Predicted y and smoothed truncated predicted y
    ypred = np.zeros(n, dtype=float)
    ytrunc = np.zeros(n, dtype=float)
    for k in harmonics:
        ypred += harmonics[k]
        if ntrunc is not None and k <= ntrunc:
            ytrunc += harmonics[k]

    return f_k, C_k, ps_k, harmonics, ypred, ytrunc

# ----------------------------------------------------------------------

# regress_field()
# time_detrend
# time_std, time_mean, etc.
