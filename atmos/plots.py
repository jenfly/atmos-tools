"""
Utility functions for plotting atmospheric data.
"""

from __future__ import division
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import xray
from atmos.utils import print_if
import atmos.data as dat
import atmos.utils as utils


# ----------------------------------------------------------------------
def degree_sign():
    """Return a degree sign for LaTeX interpreter."""
    return r'$^\circ$'


# ----------------------------------------------------------------------
def latlon_labels(vals, latlon='lat', fmt='%.0f', deg_symbol=True,
                  join_str=None):
    """Return a label string for list of latitudes or longitudes."""

    if latlon.lower() == 'lat':
        pos, neg = 'N', 'S'
    elif latlon.lower() == 'lon':
        pos, neg = 'E', 'W'
    else:
        raise ValueError('Invalid input latlon = ' + latlon)

    vals = utils.makelist(vals)
    labels = []
    for num in vals:
        if num >= 0:
            suffix = pos
            x = fmt % num
        else:
            suffix = neg
            x = fmt % abs(num)
        if deg_symbol:
            labels.append(x + degree_sign() + suffix)
        else:
            labels.append(x + suffix)
    if len(vals) == 1:
        labels = labels[0]
    elif join_str is not None:
        labels = join_str.join(labels)
    return labels


# ----------------------------------------------------------------------
def latlon_str(dim1, dim2, dimname, deg_symbol=False):
    """Return a label for range of latitudes or longitudes.

    e.g. 60E-100E, 90S-90N
    """
    dims = [dim1, dim2]
    dimstr = latlon_labels(dims, dimname, deg_symbol=deg_symbol, join_str='-')
    return dimstr


# ----------------------------------------------------------------------
def mapticks(lon_ticks, lat_ticks):
    """Add nicely formatted ticks to lat-lon map."""
    plt.xticks(lon_ticks, latlon_labels(lon_ticks,'lon'))
    plt.yticks(lat_ticks, latlon_labels(lat_ticks, 'lat'))


# ----------------------------------------------------------------------
def autoticks(axtype, axmin, axmax, width=None, nmax=8):
    """
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
    """

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
def clevels(data, cint, posneg='both', symmetric=False, omitzero=False,
            percentile=99.9):
    """
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
    omitzero : bool, optional
        Omit zero from the contour levels
    percentile : float, optional
        Percentile to use in calculating contour range.

    Returns
    -------
    clev: ndarray
        Array of contour levels
    """

    # Define max and min contour levels
    vmax = np.nanpercentile(data, percentile)
    vmin = np.nanpercentile(data, 100 - percentile)
    if symmetric:
        vmax = max(abs(vmin), abs(vmax))
        cabs = math.ceil(vmax / cint) * cint
        cmin, cmax = -cabs, cabs
    else:
        cmin = math.floor(vmin / cint) * cint
        cmax = math.ceil(vmax / cint) * cint
    if posneg == 'pos':
        cmin = 0
    elif posneg == 'neg':
        cmax = 0

    # Define contour levels, making sure to include the endpoint
    clev = np.arange(cmin, cmax + 0.1*cint, cint)

    # Omit zero, if selected
    if omitzero:
        ind = np.where(clev == 0)
        clev = np.delete(clev, ind)

    return clev


# ----------------------------------------------------------------------
def cinterval(data, n_pref=20, symmetric=False, cint_pref=[1, 2, 3, 4, 5, 10],
              percentile=99.9):
    """Return a sensible contour interval for plotting data.

    Parameters
    ----------
    data : np.ndarray or xray.DataArray
        Data to be contoured.
    n_pref : int, optional
        Preferred number of contours.  The contour interval is chosen
        so that the number of contours is close to n_pref.
    symmetric : bool, optional
        If True, then choose a contour interval appropriate to a
        color scale symmetric about zero.
    cint_pref : list, optional
        Preferred contour intervals.  The interval is c * scale,
        where scale is the appropriate order of magnitude and c is
        the value in cint_pref that gives a number of contour intervals
        closest to n_pref.
    percentile : float, optional
        Percentile (0-100) to use to calculate maximum value for data
        spread, and minimum value for data spread is 100-percentile
        percentile.

    Returns
    -------
    cint : float
    """

    if isinstance(data, xray.DataArray):
        data = data.values
    if symmetric:
        spread = 2 * np.nanpercentile(np.abs(data), percentile)
    else:
        vmin = np.nanpercentile(data, 100 - percentile)
        vmax = np.nanpercentile(data, percentile)
        spread =  vmax - vmin
    cint = spread / n_pref
    scale = 10 ** np.floor(np.log10(cint))
    cint = cint / scale
    diff = np.abs(cint - cint_pref)
    cint = cint_pref[diff.argmin()] * scale
    return cint


# ----------------------------------------------------------------------
def climits(data, symmetric=True, percentile=99.9):
    """Return colorbar limits to use for all data variables in a set.
    """

    if isinstance(data, xray.Dataset):
        data = data.to_array()

    cmin = np.nanpercentile(data, 100 - percentile)
    cmax = np.nanpercentile(data, percentile)
    if symmetric:
        cmax = max([abs(cmin), abs(cmax)])
        cmin = - cmax
    clims = (cmin, cmax)

    return clims


# ----------------------------------------------------------------------
def colorbar_symm(**kwargs):
    """Create a colorbar with limits symmetric about zero.

    Optional keyword arguments are inputs to plt.colorbar().
    """
    cb = plt.colorbar(**kwargs)
    cmax = abs(cb.boundaries).max()
    plt.clim(-cmax, cmax)


# ----------------------------------------------------------------------
def init_latlon(lat1=-90, lat2=90, lon1=0, lon2=360, fancy=True,
                resolution='c', **kwargs):
    """Initialize lon-lat plot and return as a Basemap object."""

    m = Basemap(llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2,
                resolution=resolution, **kwargs)
    m.drawcoastlines()
    xticks = autoticks('lon', lon1, lon2)
    yticks = autoticks('lat', lat1, lat2)
    if fancy:
        mapticks(xticks, yticks)
    else:
        plt.xticks(xticks)
        plt.yticks(yticks)
    plt.draw()
    return m


# ----------------------------------------------------------------------
def geobox(lat1, lat2, lon1, lon2, m=None, color='blue', linewidth=2,
           linestyle='-', label=None, axlims=(-90, 90, 0, 360)):
    """Plot a lat-lon box on a map.

    Optional input m is a Basemap object. If None, a new map is
    created with init_latlon with the lat-lon range specified in
    axlims.
    """

    if m is None:
        m = init_latlon(axlims[0], axlims[1], axlims[2], axlims[3])

    x = [lon1, lon1, lon2, lon2, lon1]
    y = [lat1, lat2, lat2, lat1, lat1]
    m.plot(x, y, latlon=True, color=color, linewidth=linewidth,
           linestyle=linestyle, label=label)
    return m


# ----------------------------------------------------------------------
def pcolor_latlon(data, lat=None, lon=None, m=None, cmap='RdBu_r',
                  axlims=None, fancy=True):
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
    axlims : 4-tuple of ints or floats, optional
        Lat-lon limits for map (lat1, lat2, lon1, lon2).  If None, then
        data range is used.
    fancy : bool, optional
        If True, init_latlon will label axes with fancy lat-lon labels.

    Returns
    -------
    m : Basemap object
    pc : plt.pcolormesh object
    cb : plt.colorbar object
    """

    if isinstance(data, xray.DataArray):
        lat = dat.get_coord(data, 'lat')
        lon = dat.get_coord(data, 'lon')
        vals = np.squeeze(data.values)
    else:
        vals = np.squeeze(data)

    # Lat-lon ranges
    if axlims is None:
        lat1, lat2 = np.floor(lat.min()), np.ceil(lat.max())
        lon1, lon2 = np.floor(lon.min()), np.ceil(lon.max())
    else:
        lat1, lat2, lon1, lon2 = axlims

    # Use a masked array so that pcolormesh displays NaNs properly
    vals_plot = np.ma.array(vals, mask=np.isnan(vals))
    x, y = np.meshgrid(lon, lat)

    if m is None:
        m = init_latlon(lat1, lat2, lon1, lon2, fancy)
    pc = m.pcolormesh(x, y, vals_plot, cmap=cmap, latlon=True)
    cb = m.colorbar()
    plt.draw()
    return m, pc, cb


# ----------------------------------------------------------------------
def contourf_latlon(data, lat=None, lon=None, clev=None, m=None, cmap='RdBu_r',
                    symmetric=True, axlims=None, fancy=True, colorbar=True,
                    **kwargs):
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
    axlims : 4-tuple of ints or floats, optional
        Lat-lon limits for map (lat1, lat2, lon1, lon2).  If None, then
        data range is used.
    fancy : bool, optional
        If True, init_latlon will label axes with fancy lat-lon labels.
    colorbar : bool, optional
        If True, include a colorbar.
    **kwargs : keyword arguments, optional
        Additional keyword arguments to plt.contourf().

    Returns
    -------
    m : Basemap object
    """

    if isinstance(data, xray.DataArray):
        lat = dat.get_coord(data, 'lat')
        lon = dat.get_coord(data, 'lon')

    if isinstance(clev, float) or isinstance(clev, int):
        # Define contour levels from selected interval spacing
        clev = clevels(data, clev, symmetric=symmetric)

    # Lat-lon ranges
    if axlims is None:
        lat1, lat2 = np.floor(lat.min()), np.ceil(lat.max())
        lon1, lon2 = np.floor(lon.min()), np.ceil(lon.max())
    else:
        lat1, lat2, lon1, lon2 = axlims

    x, y = np.meshgrid(lon, lat)

    if m is None:
        m = init_latlon(lat1, lat2, lon1, lon2, fancy)
    if clev is None:
        m.contourf(x, y, np.squeeze(data), cmap=cmap, latlon=True, **kwargs)
    else:
        m.contourf(x, y, np.squeeze(data), clev, cmap=cmap, latlon=True, **kwargs)
    if colorbar:
        m.colorbar()
    plt.draw()
    return m


# ----------------------------------------------------------------------
def contour_latlon(data, lat=None, lon=None, clev=None, m=None, colors='black',
                   linewidths=2.0, linestyles=None, axlims=None, fancy=True):
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
    linestyles :  {None, 'solid', 'dashed', 'dashdot', 'dotted'}, optional
        Line style for contour lines.
    axlims : 4-tuple of ints or floats, optional
        Lat-lon limits for map (lat1, lat2, lon1, lon2).  If None, then
        data range is used.
    fancy : bool, optional
        If True, init_latlon will label axes with fancy lat-lon labels.

    Returns
    -------
    m : Basemap object
    cs : plt.contour object
    """

    if isinstance(data, xray.DataArray):
        lat = dat.get_coord(data, 'lat')
        lon = dat.get_coord(data, 'lon')

    if isinstance(clev, float) or isinstance(clev, int):
        # Define contour levels from selected interval spacing
        clev = clevels(data, clev)

    # Lat-lon ranges
    if axlims is None:
        lat1, lat2 = np.floor(lat.min()), np.ceil(lat.max())
        lon1, lon2 = np.floor(lon.min()), np.ceil(lon.max())
    else:
        lat1, lat2, lon1, lon2 = axlimss
    x, y = np.meshgrid(lon, lat)

    if m is None:
        m = init_latlon(lat1, lat2, lon1, lon2, fancy)
    if clev is None:
        cs = m.contour(x, y, np.squeeze(data), colors=colors,
                       linewidths=linewidths, linestyles=linestyles,
                       latlon=True)
    else:
        cs = m.contour(x, y, np.squeeze(data), clev, colors=colors,
                       linewidths=linewidths, linestyles=linestyles,
                       latlon=True)
    plt.draw()
    return m, cs


# ----------------------------------------------------------------------
def init_latpres(latmin=-90, latmax=90, pmin=0, pmax=1000, topo_ps=None,
                 topo_lat=None, topo_clr='black', p_units='hPa',
                 lattick_width=None, ptick_width=None):
    """Initialize a latitude-pressure plot.

    Parameters
    ----------
    latmin, latmax, pmin, pmax : int or float, optional
        Axes limits.
    topo_ps, topo_lat : ndarray or list of floats, optional
        Surface pressure profile to plot for topography.
    topo_clr : string or mpl_color, optional
        Color to fill topographic profile.
    p_units : string, optional
        Units of pressure for y-axis and surface pressure profile.
    lattick_width, ptick_width : int or float, optional
        Spacing for latitude and pressure ticks.
    """

    xticks = autoticks('lat', latmin, latmax, lattick_width)
    yticks = autoticks('pres', pmin, pmax, ptick_width)
    plt.xlim(latmin, latmax)
    plt.xticks(xticks, latlon_labels(xticks, latlon='lat', fmt='%.0f'))
    plt.ylim(pmin, pmax)
    plt.yticks(yticks)
    plt.gca().invert_yaxis()
    plt.xlabel('Latitude')
    plt.ylabel('Pressure (' + p_units + ')')

    if isinstance(topo_ps, np.ndarray) or isinstance(topo_ps, list):
        plt.fill_between(topo_lat, pmax, topo_ps, color=topo_clr)

    plt.draw()


# ----------------------------------------------------------------------
def pcolor_latpres(data, lat=None, plev=None, init=True, cmap='RdBu_r',
                   topo=None, topo_clr='black', p_units='hPa',
                   axlims=(-90, 90, 0, 1000), lattick_width=None,
                   ptick_width=None):
    """Create pseudo-color plot of data in latitude-pressure plane.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Data to be contoured.
    lat : ndarray, optional
        Latitude (degrees).  If data is an xray.DataArray, lat is
        extracted from data['lat'].
    plev : ndarray, optional
        Pressure levels.  If data is an xray.DataArray, plev is
        extracted from data['plev'].
    init : bool, optional
        If True, initialize plot axes and topography with init_latpres().
    cmap : string or colormap object, optional
        Colormap to use.
    topo : ndarray or xray.DataArray, optional
        Topography to shade (average surface pressure in units of plev).
        If topo is an ndarray, it must be on the same latitude grid as
        data.  If topo is an xray.DataArray, its latitude grid is
        extracted from topo['lat']. Only used if init is True.
    topo_clr : string or mpl_color, optional
        Color to fill topographic profile. Only used if init is True.
    p_units : string, optional
        Units for pressure axis label.  Only used if init is True.
    axlims : 4-tuple of floats or ints
        Axis limits (latmin, latmax, pmin, pmax).  Only used if init is True.
    lattick_width, ptick_width : int or float, optional
        Spacing for latitude and pressure ticks.  Only used if init is True.

    Returns
    -------
    pc : plt.pcolormesh object
    """

    # Data to be plotted
    if isinstance(data, xray.DataArray):
        lat = dat.get_coord(data, 'lat')
        plev = dat.get_coord(data, 'plev')
        vals = np.squeeze(data.values)
    else:
        vals = np.squeeze(data)

    # Use a masked array so that pcolormesh displays NaNs properly
    vals_plot = np.ma.array(vals, mask=np.isnan(vals))

    # Pseudo-color plot of data
    y, z = np.meshgrid(lat, plev)
    pc = plt.pcolormesh(y, z, vals_plot, cmap=cmap)
    plt.colorbar()
    plt.draw()

    # Initialize plot
    if init:

        # Topography data
        if isinstance(topo, np.ndarray) or isinstance(topo, list):
            topo_ps, topo_lat = topo, lat
        elif isinstance(topo, xray.DataArray):
            topo_ps, topo_lat = topo.values, topo['lat']
        else:
            topo_ps, topo_lat = None, None

        # Initialize axes and plot topography
        latmin, latmax, pmin, pmax = axlims
        init_latpres(latmin, latmax, pmin, pmax, topo_ps=topo_ps,
                     topo_lat=topo_lat, topo_clr=topo_clr, p_units=p_units,
                     lattick_width=lattick_width, ptick_width=ptick_width)

    return pc

# ----------------------------------------------------------------------
def contourf_latpres(data, lat=None, plev=None, clev=None, init=True,
                    cmap='RdBu_r', symmetric=True, topo=None, topo_clr='black',
                    p_units='hPa', axlims=(-90, 90, 0, 1000),
                    lattick_width=None, ptick_width=None):
    """Plot filled contours of data in latitude-pressure plane.

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Data to be contoured.
    lat : ndarray, optional
        Latitude (degrees).  If data is an xray.DataArray, lat is
        extracted from data['lat'].
    plev : ndarray, optional
        Pressure levels.  If data is an xray.DataArray, plev is
        extracted from data['plev'].
    clev : float or ndarray, optional
        Contour levels (ndarray) or spacing interval (float)
    init : bool, optional
        If True, initialize plot axes and topography with init_latpres().
    cmap : string or colormap object, optional
        Colormap to use.
    symmetric : bool, optional
        Set contour levels to be symmetric about zero.
    topo : ndarray or xray.DataArray, optional
        Topography to shade (average surface pressure in units of plev).
        If topo is an ndarray, it must be on the same latitude grid as
        data.  If topo is an xray.DataArray, its latitude grid is
        extracted from topo['lat']. Only used if init is True.
    topo_clr : string or mpl_color, optional
        Color to fill topographic profile. Only used if init is True.
    p_units : string, optional
        Units for pressure axis label.  Only used if init is True.
    axlims : 4-tuple of floats or ints
        Axis limits (latmin, latmax, pmin, pmax).  Only used if init is True.
    lattick_width, ptick_width : int or float, optional
        Spacing for latitude and pressure ticks.  Only used if init is True.
    """

    # Data to be contoured
    if isinstance(data, xray.DataArray):
        lat = dat.get_coord(data, 'lat')
        plev = dat.get_coord(data, 'plev')

    # Contour levels
    if isinstance(clev, float) or isinstance(clev, int):
        # Define contour levels from selected interval spacing
        clev = clevels(data, clev, symmetric=symmetric)

    # Plot contours
    y, z = np.meshgrid(lat, plev)
    if clev is None:
        plt.contourf(y, z, np.squeeze(data), cmap=cmap)
    else:
        plt.contourf(y, z, np.squeeze(data), clev, cmap=cmap)
    plt.colorbar()

    # Initialize plot
    if init:

        # Topography data
        if isinstance(topo, np.ndarray) or isinstance(topo, list):
            topo_ps, topo_lat = topo, lat
        elif isinstance(topo, xray.DataArray):
            topo_ps, topo_lat = topo.values, topo['lat']
        else:
            topo_ps, topo_lat = None, None

        # Initialize axes and plot topography
        latmin, latmax, pmin, pmax = axlims
        init_latpres(latmin, latmax, pmin, pmax, topo_ps=topo_ps,
                     topo_lat=topo_lat, topo_clr=topo_clr, p_units=p_units,
                     lattick_width=lattick_width, ptick_width=ptick_width)

    plt.draw()


# ----------------------------------------------------------------------
def contour_latpres(data, lat=None, plev=None, clev=None, init=True,
                    colors='black', topo=None, topo_clr='black', p_units='hPa',
                    axlims=(-90, 90, 0, 1000), lattick_width=None,
                    ptick_width=None, omitzero=False, zerolinewidth=2,
                    contour_kw={}):
    """
    Plot contour lines in latitude-pressure plane

    Parameters
    ----------
    data : ndarray or xray.DataArray
        Data to be contoured.
    lat : ndarray, optional
        Latitude (degrees).  If data is an xray.DataArray, lat is
        extracted from data['lat'].
    plev : ndarray, optional
        Pressure levels.  If data is an xray.DataArray, plev is
        extracted from data['plev'].
    clev : float or ndarray, optional
        Contour levels (ndarray) or spacing interval (float)
    init : bool, optional
        If True, initialize plot axes and topography with init_latpres().
    colors: string or mpl_color, optional
        Contour line color.
    topo : ndarray or xray.DataArray, optional
        Topography to shade (average surface pressure in units of plev).
        If topo is an ndarray, it must be on the same latitude grid as
        data.  If topo is an xray.DataArray, its latitude grid is
        extracted from topo['lat']. Only used if init is True.
    topo_clr : string or mpl_color, optional
        Color to fill topographic profile. Only used if init is True.
    p_units : string, optional
        Units for pressure axis label.  Only used if init is True.
    axlims : 4-tuple of floats or ints
        Axis limits (latmin, latmax, pmin, pmax).  Only used if init is True.
    lattick_width, ptick_width : int or float, optional
        Spacing for latitude and pressure ticks.  Only used if init is True.
    omitzero : bool, optional
        If True, omit zero contour.
    zerolinewidth : int or float, optional
        Include zero contour with specified line width.
    contour_kw : dict, optional
        Dict of additional keyword arguments to plt.contour().

    Returns
    -------
    cs : plt.contour object (non-zero contours only)
    """

    # Data to be contoured
    if isinstance(data, xray.DataArray):
        lat = dat.get_coord(data, 'lat')
        plev = dat.get_coord(data, 'plev')

    # Initialize plot
    if init:

        # Topography data
        if isinstance(topo, np.ndarray) or isinstance(topo, list):
            topo_ps, topo_lat = topo, lat
        elif isinstance(topo, xray.DataArray):
            topo_ps, topo_lat = topo.values, topo['lat']
        else:
            topo_ps, topo_lat = None, None

        # Initialize axes and plot topography
        latmin, latmax, pmin, pmax = axlims
        init_latpres(latmin, latmax, pmin, pmax, topo_ps=topo_ps,
                     topo_lat=topo_lat, topo_clr=topo_clr, p_units=p_units,
                     lattick_width=lattick_width, ptick_width=ptick_width)

    # Contour levels
    if isinstance(clev, float) or isinstance(clev, int):
        # Define contour levels from selected interval spacing
        clev = clevels(data, clev, omitzero=omitzero)

    # Plot contours
    y, z = np.meshgrid(lat, plev)
    if clev is None:
        cs = plt.contour(y, z, np.squeeze(data), colors=colors, **contour_kw)
    else:
        cs = plt.contour(y, z, np.squeeze(data), clev, colors=colors, **contour_kw)

    # Zero contour
    if not omitzero and zerolinewidth > 0:
        plt.contour(y, z, np.squeeze(data), 0, colors=colors,
                    linewidths=zerolinewidth, **contour_kw)

    plt.draw()
    return cs


# ----------------------------------------------------------------------
def stipple_pts(pts_mask, xname, yname, xsample=1, ysample=1, ax=None,
                marker='+', color='k', alpha=0.25, markersize=6,
                markeredgewidth=1.5, **kwargs):
    """Plot points to stipple a figure.

    Parameters
    ----------
    pts_mask: xray.DataArray
        Masking array of points to exclude from stippling.
    xname, yname : str
        Name of x and y dimensions in pts_mask.
    xsample, ysample : int
        Sub-sampling of x- and y- dimensions.
    ax : plt.axes object
        Axes to plot on.  If None, then use current axes.

    Remaining parameters are keyword arguments to plt.plot() to specify
    format for plotting the stipple points.
    """

    # Get grid points
    x = dat.get_coord(pts_mask, xname)
    y = dat.get_coord(pts_mask, yname)
    xgrid, ygrid = np.meshgrid(x, y)

    # Mask out points to be excluded
    if not xgrid.shape == pts_mask.shape:
        pts_mask = pts_mask.T
    xpts = np.ma.masked_array(xgrid, mask=pts_mask)
    ypts = np.ma.masked_array(ygrid, mask=pts_mask)

    # Sub-sample
    xpts = xpts[::ysample, ::xsample]
    ypts = ypts[::ysample, ::xsample]

    # Plot stippling
    if ax is not None:
        plt.sca(ax)
    plt.plot(xpts, ypts, marker=marker, color=color, alpha=alpha,
             markersize=markersize, markeredgewidth=markeredgewidth, **kwargs)

    return None


# ----------------------------------------------------------------------
# TO DO:
#
# mapaxes(m, axlims, xticks, yticks) - adjust limits, ticks and tick labels of
#         existing map (might have to create new basemap object to do this)
