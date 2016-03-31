import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import pandas as pd

import atmos as atm
import utils # monsoon-onset

mpl.rcParams['font.size'] = 10

# ----------------------------------------------------------------------

onset_nm = 'CHP_MFC'
years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/analysis/'

varnms = ['U200', 'T200']

datafiles = collections.OrderedDict()
filestr = datadir + 'merra_%s_dailyrel_%s_%d.nc'
for nm in varnms:
    datafiles[nm] = [filestr % (nm, onset_nm, yr) for yr in years]

# ----------------------------------------------------------------------
# Read data

data = collections.OrderedDict()
for nm in datafiles:
    var, onset, retreat = utils.load_dailyrel(datafiles[nm])
    data[nm] = var

# ----------------------------------------------------------------------
# Test atm.regress_field

day = -60

# 1-d timeseries
var = data['U200'].sel(dayrel=day)
ts = atm.mean_over_geobox(var, 10, 30, 60, 100)
ts_reg = atm.Linreg(onset, ts)
ts_reg2 = atm.regress_field(ts, onset)
print(ts_reg.r, ts_reg2.r.values)
print(ts_reg.slope, ts_reg2.m.values)
print(ts_reg.p, ts_reg2.p.values)

# x-y data
regdays = [-60, -30, 0, 30, 60]
plotdays = [-60, -30]
clev_r = np.arange(-1.0, 1.01, 0.05)
for nm in varnms:
    print(nm)
    var = data[nm].sel(dayrel=regdays)
    reg_daily = atm.regress_field(var, onset, axis=0)
    for day in plotdays:
        reg = reg_daily.sel(dayrel=day)
        title = '%s day %d vs. Onset ' % (var.name, day)
        cint_m = atm.cinterval(reg.m)
        clev_m = atm.clevels(reg.m, cint_m, symmetric=True)
        plt.figure(figsize=(11, 8))
        plt.subplot(1, 2, 1)
        atm.contourf_latlon(reg['r'], clev=clev_r, cmap='RdBu_r')
        plt.title(title + ' - Corr Coeff')
        plt.subplot(1, 2, 2)
        atm.contourf_latlon(reg['m'], clev=clev_m, cmap='RdBu_r')
        plt.title(title + ' - Reg Coeff')
