'''
Utility functions for plotting atmospheric data.
'''

import math
import numpy as np
import matplotlib.pyplot as plt
from atmos.utils import print_if
from atmos.exceptions import InputError
'''
TO DO:
clevels - omit zero option

latlon_ticks

contour_latpres - format dictionaries for contours and topography,
    - zero contours treated separately - omit or make different color/width
'''

# ----------------------------------------------------------------------
def autotick(axtype, axmin, axmax, width=None):
    '''
    Return an array of sensible automatic tick positions.

    Parameters
    ----------
    axtype : {'lon', 'lat', 'pres'}
        Type of axis - longitude, latitude or pressure level
    axmin, axmax : float or int
        Axis limits
    width : float, optional
        Spacing of ticks.  Omit to let the function set an auto value.

    Returns
    -------
    ticks : ndarray
        Array of tick positions
    '''

    if not width:
        # Set the width between ticks
        diff = axmax - axmin
        if axtype.lower() == 'lon' or axtype.lower() == 'lat':
            if diff > 270:
                width = 60
            elif diff > 120:
                width = 30
            else:
                width = 15
        elif axtype.lower() == 'pres':
            if diff > 500:
                width = 200
            else:
                width = 100
        else:
            raise InputError('Invalid axtype: ' + axtype)

    # Set the ticks
    n1 = math.ceil(float(axmin)/width)
    n2 = math.floor(float(axmax)/width)
    ticks = np.arange(n1*width, (n2+0.1)*width, width)
    return ticks


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
