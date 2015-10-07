"""Testing load_concat"""

import numpy as np
import matplotlib.pyplot as plt
import xray

import atmos.plots as ap
import atmos.data as dat
import atmos.utils as utils
from atmos.data import load_concat

# ----------------------------------------------------------------------
# Read data
url = ('http://goldsmr3.sci.gsfc.nasa.gov/opendap/MERRA_MONTHLY/'
'MAIMCPASM.5.2.0/2014/MERRA300.prod.assim.instM_3d_asm_Cp.2014%02d.hdf')
paths = [url % 1, url % 2, url % 3 ]

var_ids = ['U', 'V', 'T']

lon1, lon2 = 0, 100
lat1, lat2 = -45, 45

ds1 = load_concat(paths, var_ids)
ds2 = load_concat(paths, var_ids, subset1=('lat', lat1, lat2),
                  subset2=('lon', lon1, lon2))
ds3 = load_concat(paths, 'PS', subset1=('lat', lat1, lat2),
                  subset2=('lon', lon1, lon2))
u1 = load_concat(paths, 'U')
u2 = load_concat(paths[0], 'U')
