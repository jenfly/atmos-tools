'''
Utility functions for plotting atmospheric data.
'''

import math
import numpy as np
import matplotlib.pyplot as plt
from atmos.utils import print_if

'''
TO DO:
clevels - omit zero option

latlon_ticks

contour_latpres - format dictionaries for contours and topography,
    - zero contours treated separately - omit or make different color/width
'''


# ----------------------------------------------------------------------
def clevels(data, cint, posneg='both', symmetric=False):
    '''
    Return array of contour levels spaced by a given interval.

    Parameters
    ----------
    data : ndarray
        Data to be contoured
    cint : float
        Spacing of contour intervals
    posneg : {'both', 'pos', 'neg'}, optional
        Return all contours or only pos/neg
    symmetric : bool, optional
        Return contour levels symmetric about zero

    Returns
    -------
    clev: ndarray
        Array of contour levels
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
    clev = np.arange(cmin, cmax + 0.1*cint, cint)
    return clev

# ----------------------------------------------------------------------
def contour_latpres(lat, pres, data, clev, c_color='black', topo=None):
    '''
    Plot contour lines in latitude-pressure plane

    Parameters
    ----------
    lat : ndarray
        Latitude (degrees)
    pres : ndarray
        Pressure levels (hPa)
    data : ndarray
        Data to be contoured
    clev : float or ndarray
        Contour levels (ndarray) or spacing interval (float)
    c_color: string or mpl_color, optional
        Contour line color
    topo : ndarray, optional
        Topography to shade (average surface pressure in units of pres)
    '''

    # Contour levels
    if not isinstance(clev, list) and not isinstance(clev, np.ndarray):
        clev = clevels(data, clev)

    # Grid for plotting
    y, z = np.meshgrid(lat, pres)

    # Plot contours
    pmin, pmax = 0, 1000
    if isinstance(topo, np.ndarray) or isinstance(topo, list):
        plt.fill_between(lat, pmax, topo, color='black')
    plt.contour(y, z, data, clev, colors=c_color)
    plt.ylim(pmin, pmax)
    plt.gca().invert_yaxis()
    plt.xticks(np.arange(-90, 90, 30))
    plt.xlabel('Latitude')
    plt.ylabel('Pressure (hPa)')
    plt.draw()
