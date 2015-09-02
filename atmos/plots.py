'''
Utility functions for plotting atmospheric data.
'''

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import xray
from atmos.utils import print_if

'''
TO DO:
clevels - omit zero option

mapticks - change tick label formatting to 0-360E rather than 180W to 180E
        -> Define a function to return the formatted ticks and use fmt
           keyword argument in m.drawmeridians

mapaxes(m, axlims, xticks, yticks) - adjust limits, ticks and tick labels of
        existing map (might have to create new basemap object to do this)

contour_latpres - format dictionaries for contours and topography,
    - zero contours treated separately - omit or make different color/width
'''

# ----------------------------------------------------------------------
def autoticks(axtype, axmin, axmax, width=None, nmax=8):
    '''
    Return an array of sensible automatic tick positions for geo data.

    Parameters
    ----------
    axtype : {'pres', 'lat', 'lon'}
        Type of axis - longitude, latitude or pressure level
    axmin, axmax : float or int
        Axis limits
    width : float, optional
        Spacing of ticks.  Omit to let the function set an auto value.
    nmax : int, optional
        Maximum number of ticks.  This is ignored if width is specified.

    Returns
    -------
    ticks : ndarray
        Array of tick positions
    '''

    if not width:
        # Set the width between ticks
        diff = axmax - axmin
        if axtype.lower() == 'lon' or axtype.lower() == 'lat':
            wlist = [10, 15, 30, 60]
        elif axtype.lower() == 'pres':
            wlist = [50, 100, 200]
        else:
            raise ValueError('Invalid axtype: ' + axtype)

        for w in wlist:
            n1 = math.ceil(float(axmin)/w)
            n2 = math.floor(float(axmax)/w)
            ntick = n2 - n1 + 1
            if ntick <= nmax:
                width = w
                break
    else:
        # Use the width specified in input
        n1 = math.ceil(float(axmin)/width)
        n2 = math.floor(float(axmax)/width)

    # Set the ticks
    ticks = np.arange(n1*width, (n2+0.1)*width, width)
    return ticks


# ----------------------------------------------------------------------
def mapticks(m, xticks, yticks, labels=['left', 'bottom'],
             gridlinewidth=0.0):
    """Add nicely formatted ticks to basemap."""

    label_dict = {'left' : 0, 'right' : 1, 'top' : 2, 'bottom' : 3}
    lvec = [0, 0, 0, 0]
    for nm in labels:
        lvec[label_dict[nm]] = 1

    plt.xticks(xticks, [])
    plt.yticks(yticks, [])
    m.drawmeridians(xticks, labels=lvec, labelstyle='E',
                    linewidth=gridlinewidth)
    m.drawparallels(yticks, labels=lvec, labelstyle='N/S',
                    linewidth=gridlinewidth)


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
def init_latlon(lat1=-90, lat2=90, lon1=0, lon2=360, labels=['left', 'bottom'],
                gridlinewidth=0.0):
    """Initialize lon-lat plot."""

    m = Basemap(llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
    m.drawcoastlines()
    xticks = autoticks('lon', lon1, lon2)
    yticks = autoticks('lat', lat1, lat2)
    mapticks(m, xticks, yticks, labels=labels, gridlinewidth=gridlinewidth)
    plt.draw()
    return m


# ----------------------------------------------------------------------
def pcolor_latlon(data, lat=None, lon=None, m=None, cmap='RdBu_r',
                  axlims=(-90, 90, 0, 360)):
    """Create a pseudo-color plot of geo data.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Data to be plotted.
    lat, lon : ndarray, optional
        Latitude and longitude arrays.  Only used if data is an ndarray.
        If data is an xray.DataArray then lat = data['lat'] and
        lon = data['lon']
    m : Basemap object, optional
        Basemap to plot on.  If omitted, then a map is created with
        init_latlon().
    cmap : string or colormap object, optional
        Colormap to use.
    axlims : 4-tuple of ints or floats
        Lat-lon limits for map.

    Returns
    -------
    m : Basemap object
    """

    if isinstance(data, xray.DataArray):
        lat, lon = data['lat'], data['lon']

    x, y = np.meshgrid(lon, lat)
    lat1, lat2, lon1, lon2 = axlims

    if m is None:
        m = init_latlon(lat1, lat2, lon1, lon2)
    m.pcolormesh(x, y, data, cmap=cmap, latlon=True)
    m.colorbar()
    plt.draw()
    return m


# ----------------------------------------------------------------------
def contourf_latlon(data, lat=None, lon=None, clev=None, m=None, cmap='RdBu_r',
                    symmetric=True, axlims=(-90, 90, 0, 360)):
    """Create a filled contour plot of geo data.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Data to be plotted.
    lat, lon : ndarray, optional
        Latitude and longitude arrays.  Only used if data is an ndarray.
        If data is an xray.DataArray then lat = data['lat'] and
        lon = data['lon']
    clev : scalar or array of ints or floats, optional
        Contour spacing (scalar) or contour levels (array).
    m : Basemap object, optional
        Basemap to plot on.  If omitted, then a map is created with
        init_latlon().
    cmap : string or colormap object, optional
        Colormap to use.
    symmetric : bool, optional
        Set contour levels to be symmetric about zero.
    axlims : 4-tuple of ints or floats
        Lat-lon limits for map.
    """

    if isinstance(data, xray.DataArray):
        lat, lon = data['lat'], data['lon']

    if isinstance(clev, float) or isinstance(clev, int):
        # Define contour levels from selected interval spacing
        clev = clevels(data, clev, symmetric=symmetric)

    lat1, lat2, lon1, lon2 = axlims
    x, y = np.meshgrid(lon, lat)

    if m is None:
        m = init_latlon(lat1, lat2, lon1, lon2)
    if clev is None:
        m.contourf(x, y, data, cmap=cmap, latlon=True)
    else:
        m.contourf(x,y, data, clev, cmap=cmap, latlon=True)
    m.colorbar()
    plt.draw()
    return m


# ----------------------------------------------------------------------
def contour_latlon(data, lat=None, lon=None, clev=None, m=None, colors='black',
                   linewidths=2.0, axlims=(-90, 90, 0, 360)):
    """Create a contour line plot of geo data.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Data to be plotted.
    lat, lon : ndarray, optional
        Latitude and longitude arrays.  Only used if data is an ndarray.
        If data is an xray.DataArray then lat = data['lat'] and
        lon = data['lon']
    clev : scalar or array of ints or floats, optional
        Contour spacing (scalar) or contour levels (array).
    m : Basemap object, optional
        Basemap to plot on.  If omitted, then a map is created with
        init_latlon().
    colors : string or mpl_color, optional
        Contour line color(s).
    linewidths : int or float, optional
        Line width for contour lines
    axlims : 4-tuple of ints or floats
        Lat-lon limits for map.

    """

    if isinstance(data, xray.DataArray):
        lat, lon = data['lat'], data['lon']
        
    if isinstance(clev, float) or isinstance(clev, int):
        # Define contour levels from selected interval spacing
        clev = clevels(data, clev)

    lat1, lat2, lon1, lon2 = axlims
    x, y = np.meshgrid(lon, lat)

    if m is None:
        m = init_latlon(lat1, lat2, lon1, lon2)
    if clev is None:
        m.contour(x, y, data, colors=colors, linewidths=linewidths,
                  latlon=True)
    else:
        m.contour(x,y, data, clev, colors=colors, linewidths=linewidths,
                  latlon=True)
    plt.draw()
    return m


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
