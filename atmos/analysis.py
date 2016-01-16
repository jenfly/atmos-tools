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
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

import atmos.utils as utils
#import atmos.xrhelper as xr

# ======================================================================
# SPECTRAL ANALYSIS
# ======================================================================

class Fourier:
    def __init__(self, y, dt=1.0, t=None, axis=0, time_units=None,
                 data_name=None):
        """Return Fourier transform of a timeseries.

        Uses the numpy FFT function for real-valued inputs,
        numpy.fft.rfft().

        Parameters
        ----------
        y : ndarray
            N-D array of timeseries data.
        dt : float, optional
            Time spacing of data.
        axis : int, optional
            Time dimension to use for FFT.
        t : ndarray
            Array of times (e.g. datetimes for plotting timeseries).
        time_units : str, optional
            Time units.
        data_name : str, optional
            Name of timeseries data.

        Returns
        -------
        self : Fourier object
            The Fourier object has the following data attributes:
              t, tseries : ndarray
                Time values and data from input timeseries
              f_k, tau_k : ndarray
                Frequencies and periods in Fourier transform.
              C_k : complex ndarray
                Fourier coefficients.
              ps_k : ndarray
                Power spectral density at each frequency.

            And it has the following methods:
              smooth() : Smooth a timeseries with truncated FFT.
              harmonic() : Return the k'th harmonic of the FFT.
              Rsquared() : Return the Rsquared values of the FFT.
        """

        # Make sure we're working with an ndarray and not a DataArray
        if isinstance(y, xray.DataArray):
            y = y.values

        self.attrs = {'data_name' : data_name, 'time_units' : time_units,
                      'dt' : dt}
        dims = y.shape
        n = dims[axis]
        if t is None:
            t = dt * np.arange(n)
        self.t = t
        self.tseries = y
        self.n = n

        # Fourier frequencies and coefficients
        self.f_k = np.fft.rfftfreq(n, dt)
        self.C_k = np.fft.rfft(y, axis=axis)
        self.axis = axis

        # Periods corresponding to Fourier frequencies
        self.tau_k = np.concatenate(([np.nan], 1/self.f_k[1:]))

        # Power spectral density
        ps_k = 2 * np.abs(self.C_k / n)**2
        self.ps_k = ps_k


    def __repr__(self):
        def var_str(name, x):
            width = 10
            return '  %s %s\n'  % (name.ljust(width),str(x.shape))

        s = 'Attributes\n' + str(self.attrs)
        s = s + '\n  Axis: %d\n  n: %d\n' % (self.axis, self.n)
        s = s + '\nData\n'
        s = s + (var_str('t', self.t) + var_str('tseries', self.tseries) +
                 var_str('f_k', self.f_k) + var_str('tau_k', self.tau_k) +
                 var_str('C_k', self.C_k) + var_str('ps_k', self.ps_k))

        return s

    def smooth(self, kmax):
        """Return a smooth timeseries from the FFT truncated at kmax."""
        n = self.n
        ax = self.axis
        C_k = self.C_k
        C_k = np.split(C_k, [kmax + 1], axis=ax)[0]
        ysmooth = np.fft.irfft(C_k, n, axis=ax)
        return ysmooth

    def harmonic(self, k):
        """Return the k'th Fourier harmonic of the timeseries."""
        if k == 0:
            y = self.smooth(k)
        else:
            y = self.smooth(k) - self.smooth(k - 1)
        return y

    def Rsquared(self):
        """Return the coefficients of determination of the FFT.

        The sum of the Rsq values up to and including the k-th Fourier
        harmonic gives the amount of variance explained by those
        harmonics.
        """
        axis = self.axis
        var = np.var(self.tseries, axis=axis)
        ps_k = self.ps_k

        # The k=0 harmonic (i.e. constant function) does not contribute
        # to the variance in the timeseries.
        if axis == 1:
            var = np.expand_dims(var, axis)
            ps_k[:, 0] = 0.0
        elif axis ==0:
            ps_k[0] = 0.0
        else:
            raise ValueError('Invalid axis ' + str(axis))
        Rsq = ps_k / var

        return Rsq


# ----------------------------------------------------------------------
def fourier_from_scratch(y, dt=1.0, ntrunc=None):
    """Calculate Fourier transform from scratch and smooth a timeseries.

    Parameters
    ----------
    y : ndarray
        1-D array of timeseries data.
    dt : float, optional
        Time spacing of data.
    ntrunc : int, optional
        Maximum harmonics to include in smoothed output.

    Returns
    -------
    f_k : ndarray
        Frequencies in Fourier transform.
    C_k : complex ndarray
        Fourier coefficients.
    ps_k : ndarray
        Power spectral density at each frequency.
    harmonics : dict of ndarrays
        Timeseries of harmonics corresponding to each Fourier frequency.
    ypred : ndarray
        Timeseries of data predicted by Fourier coefficients, to
        check that the reverse Fourier transform matches the input
        timeseries.
    ytrunc : ndarray
        Smoothed timeseries of data predicted by the Fourier harmonics
        truncated at maximum frequency f_k where k = ntrunc.

    Reference
    ---------
    Wilks, D. S. (2011). Statistical Methods in the Atmospheric Sciences.
    International Geophysics (Vol. 100).
    """

    n = len(y)
    nhalf = n // 2
    t = np.arange(1, n+1)
    omega1 = 2 * np.pi / n

    # Fourier transform and harmonics
    # --------------------------------
    # Calculate according to equations 9.62-9.65 in Wilks (2011) and
    # rescale to be consistent with scaling in the output from
    # numpy's FFT function.
    Ak = np.zeros(nhalf + 1, dtype=float)
    Bk = np.zeros(nhalf + 1, dtype=float)
    Ak[0] = np.sum(y) / 2.0
    Bk[0] = 0.0
    harmonics = {0 : np.mean(y)}
    for k in range(1, nhalf + 1):
        omega = k * omega1
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
def fourier_smooth(data, kmax, axis=0):
    """Return data smoothed with Fourier series truncated at kmax.

    Parameters
    ----------
    data : ndarray
        Data to smooth.
    kmax : int
        Maximum Fourier harmonic to include.
    axis : int, optional
        Dimension to compute Fourier transform and smoothing.

    Returns
    -------
    data_out : ndarray
        Smoothed data.
    Rsq : ndarray
        Coefficient of determination for the smoothed data,
        i.e. fraction of total variance accounted for by the
        smoothed data.
    """

    ft = Fourier(data, axis=axis)
    data_out = ft.smooth(kmax)
    Rsq = ft.Rsquared()
    Rsq = np.split(Rsq, [kmax + 1], axis=axis)[0]
    Rsq = np.sum(Rsq, axis=axis)
    return data_out, Rsq


# ======================================================================
# LINEAR REGRESSION AND CORRELATIONS
# ======================================================================

class Linreg:
    def __init__(self, x, y):
        """Return least-squares regression line.

        See scipy.stats.linregress for details."""
        reg = scipy.stats.linregress(x, y)
        self.slope, self.intercept, self.r, self.p, self.stderr = reg
        self.x, self.y = x, y

    def __repr__(self):
        s = 'Slope: ' + str(self.slope)
        s = s + '\nIntercept: ' + str(self.intercept)
        s = s + '\nCorrelation coefficient: ' + str(self.r)
        s = s + '\np-value: ' + str(self.p)
        s = s + '\nStandard error: ' + str(self.stderr)
        return s

    def predict(self, x):
        """Return predicted y values from linear regression."""
        ypred = np.polyval([self.slope, self.intercept], x)
        return ypred

    def plot(self, scatter_clr='b', scatter_sym='o', scatter_size=5,
             line_clr='r', line_width=1):
        plt.plot(self.x, self.y, color=scatter_clr, marker=scatter_sym,
                 markersize=scatter_size, linestyle='none')
        ypred = self.predict(self.x)
        plt.plot(self.x, ypred, color=line_clr, linewidth=line_width)


def corr_matrix(df, incl_index=False):
    """Return correlation coefficients and p-values between data pairs.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    incl_index : bool, optional
        If True, include the index in the pairs of correlation calculations.

    Returns
    -------
    corr : dict of DataFrames
        Correlation coefficients (corr['r']) and p-values (corr['p']) between
        each pair of columns in df.
    """

    if incl_index:
        df[df.index.name] = df.index

    cols = df.columns
    n = len(cols)
    r = np.nan * np.ones((n, n))
    p = np.nan * np.ones((n, n))

    for i in range(n):
        for j in range(i + 1):
            r[i, j], p[i, j] = scipy.stats.pearsonr(df[cols[i]], df[cols[j]])
            r[j, i], p[j, i] = r[i, j], p[i, j]

    corr = {}
    corr['r'] = pd.DataFrame(r, index=cols, columns=cols)
    corr['p'] = pd.DataFrame(p, index=cols, columns=cols)

    return corr


# ----------------------------------------------------------------------
def scatter_matrix(data, corr_fmt='%.2f', corr_pos=(0.1, 0.85), figsize=(16,10),
                   suptitle=None, incl_p=False, p_pos=(0.1, 0.75)):
    """Matrix of scatter plots with correlation coefficients annotated.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot.
    corr_fmt : str, optional
        Formatting for annotation with correlation coefficient.
    corr_pos : tuple, optional
        x, y position for annotation (dimensionless units from 0-1).
    figsize : tuple, optional
        Figure size.
    suptitle : str, optional
        Supertitle to go above subplots.
    incl_p : bool, optional
        If True, include p-value in annotation.
    p_pos : tuple, optional
        x, y position for p-value annotation. Only used if incl_p is True.
    """

    # Correlation coefficients between columns
    corr = corr_matrix(data)
    data_corr = corr['r'].as_matrix()
    data_p = corr['p'].as_matrix()

    # Matrix of scatter plots
    ax = pd.scatter_matrix(data, figsize=figsize)

    # Annotate with correlation coefficients and p-values
    if not corr_fmt.startswith('%'):
        corr_fmt = '%' + corr_fmt
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            utils.text(corr_fmt % data_corr[i, j], corr_pos, ax=ax[i, j],
                       fontweight='bold', color='black')
            if incl_p:
                p = data_p[i, j]
                if p < 0.01:
                    p_str = 'p=%.1e' % p
                else:
                    p_str = 'p=%.2f' % p
                utils.text(p_str, p_pos, ax=ax[i, j], color='black')

    plt.draw()
    if suptitle is not None:
        plt.suptitle(suptitle)



# ----------------------------------------------------------------------
# regress_field()
# time_detrend
# time_std, time_mean, etc.
