'''
Utility functions for plotting atmospheric data.
'''

import math
import numpy as np
import matplotlib.pyplot as plt
from atmos.utils import print_if

# ----------------------------------------------------------------------
def clevels(data, cint, posneg='both', symmetric=False):
    '''Returns vector of contour levels spaced by cint.

    posneg = 'both', 'pos', or 'neg' to return all contours or only pos/neg
    symmetric = True to return contour levels symmetric about zero
    '''

    # Define max and min contour levels
    if symmetric:
        cabs = math.ceil(abs(data).max() / cint) * cint
        cmin, cmax = -cabs, cabs
    else:
        cmin = math.floor(data.min() / cint) * cint
        cmax = math.ceil(data.max() / cint) * cint
    if posneg == 'pos':
        cmin = 0
    elif posneg == 'neg':
        cmax = 0

    # Define contour levels, making sure to include the endpoint
    return np.arange(cmin, cmax + 0.1*cint, cint)
