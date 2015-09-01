"""
Utility functions to for atmospheric data wrangling / preparation.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import xray


def get_topo(lon, lat, datafile='data/topo/ncep2_ps.nc')
