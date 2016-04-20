"""
Some general purpose utility functions used by other modules in this package.
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import collections
from datetime import datetime
import os
from PyPDF2 import PdfFileMerger, PdfFileReader

# ======================================================================
# PRINTING & STRING FORMATTING
# ======================================================================

# ----------------------------------------------------------------------
def print_if(msg, condition, printfunc=None):
    """ Print msg if condition is True"""
    if condition:
        if printfunc is not None:
            printfunc(msg)
        else:
            print(msg)

def format_num(x, ndecimals=2, plus_sym=False):
    """Return a nicely formatted number in .f or .e format."""
    fmt = '%.' + str(ndecimals)
    if abs(x) < 10**(-ndecimals):
        fmt = fmt + 'e'
    else:
        fmt = fmt + 'f'
    if plus_sym and x > 0:
        fmt = '+' + fmt
    return fmt % x

# ======================================================================
# FILES & DIRECTORIES
# ======================================================================

# ----------------------------------------------------------------------
def homedir(options=['/home/jennifer/', '/home/jwalker/']):
    """Return home directory for this computer."""

    home = None
    for h in options:
        if os.path.isdir(h):
            home = h
    if home is None:
        raise ValueError('Home directory not found in list of options.')
    return home


# ----------------------------------------------------------------------
def pdfmerge(filenames, outfile, delete_indiv=False):
    """Merge PDF files into a single file."""

    # Merge the files
    merger = PdfFileMerger()
    for filename in filenames:
        merger.append(PdfFileReader(file(filename, 'rb')))
    merger.write(outfile)

    # Delete the individual files
    if delete_indiv:
        for filename in filenames:
            os.remove(filename)


# ======================================================================
# PLOTS
# ======================================================================

def legend_2ax(ax1, ax2, **kwargs):
    """Create a combined legend for two y-axes."""
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, **kwargs)
    return None


def text(s, pos, ax=None, dimensionless=True, **kwargs):
    """Add text to axes.

    Parameters
    ----------
    s : str
        Text to add.
    pos : {'topleft', 'topmid', 'topright', 'midleft', 'midmid', 'midright',
           'bottomleft', 'bottommid', 'bottomright'},
           or 2-tuple of floats/ints
        String describing position for text annotation, or a tuple (x, y)
        with the exact position.
    ax : plt.AxesSubplot object, optional
        Axes to use.  If omitted, currently active axes are used.
    dimensionless : bool, optional
        If True, use dimensionless position with x, y between 0 and 1.
        If False, specify position based on data values of axes. In
        this case, pos must be an (x, y) tuple.
        If pos is a string, then dimensionless is set to True.
    kwargs : other keyword arguments
        See plt.text documentation for other keyword arguments.

    Returns
    -------
    txt : plt.text.Text object
    """

    xleft, xmid, xright = 0.1, 0.5, 0.9
    ytop, ymid, ybottom = 0.85, 0.5, 0.1

    params = {'topleft' : (xleft, ytop, 'left', 'top'),
              'topmid' : (xmid, ytop, 'center', 'top'),
              'topright' : (xright, ytop, 'right', 'top'),
              'midleft' : (xleft, ymid, 'left', 'center'),
              'midmid' : (xmid, ymid, 'center', 'center'),
              'midright' : (xright, ymid, 'right', 'center'),
              'bottomleft' : (xleft, ybottom, 'left', 'bottom'),
              'bottommid' : (xmid, ybottom, 'center', 'bottom'),
              'bottomright' : (xright, ybottom, 'right', 'bottom')}

    if isinstance(pos, str):
        if pos.lower() in params.keys():
            x, y, horiz, vert = params[pos.lower()]
            kwargs['horizontalalignment'] = horiz
            kwargs['verticalalignment'] = vert
            dimensionless = True
        else:
            raise ValueError('Invalid pos ' + pos)
    elif isinstance(pos, tuple) and len(pos) == 2:
        x, y = pos
    else:
        raise ValueError('Invalid pos %s.  Valid options are %s' %
               (str(pos), ', '.join(params.keys())))

    if ax is None:
        ax = plt.gca()
    if dimensionless:
        kwargs['transform'] = ax.transAxes

    if not kwargs:
        txt = ax.text(x, y, s)
    else:
        txt = ax.text(x, y, s, **kwargs)
    plt.draw()

    return txt


# ----------------------------------------------------------------------
def subplot_index(nrow, ncol, k, kmin=1):
    """Return the i, j index for the k-th subplot."""
    i = 1 + (k - kmin) // ncol
    j = 1 + (k - kmin) % ncol
    if i > nrow:
        raise ValueError('k = %d exceeds number of rows' % k)
    return i, j


# ----------------------------------------------------------------------
def fmt_subplot(nrow, ncol, i, ax=None, xlabel=None, xticks=None,
                xticklabels=None, ylabel=None, yticks=None, yticklabels=None):
    """Format subplots with consistent ticks and label only outer subplots.
    """

    row = i // ncol + 1
    col = (i - 1) % ncol + 1

    if ax is None:
        plt.subplot(nrow, ncol, i)
        ax = plt.gca()

    # Consistent ticks
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Labels
    if row == nrow:
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
    else:
        ax.set_xlabel('')
        ax.set_xticklabels([])
    if col == 1:
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels)
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])

    return None


# ----------------------------------------------------------------------
def fmt_axlabels(axtype, label, ax=None, **opts):
    """Format axis label and tick labels with selected options.

    Parameters
    ----------
    axtype : {'x', 'y'}
        Which axis to format.
    label : str
        Axis label.
    ax : plt.axes() object
        Axes object to format.  If None, then ax = plt.gca().
    **opts : keyword arguments
        e.g. color, alpha
    """

    if ax is None:
        ax = plt.gca()
    if axtype.lower() == 'x':
        ax.set_xlabel(label, **opts)
        ticks = ax.get_xticklabels()
    else:
        ax.set_ylabel(label, **opts)
        ticks = ax.get_yticklabels()

    if 'color' in opts:
        for t1 in ticks:
            t1.set_color(opts['color'])
    if 'alpha' in opts:
        for t1 in ticks:
            t1.set_alpha(opts['alpha'])
    plt.draw()

    return None

# ----------------------------------------------------------------------
class FigGroup:
    def __init__(self, nrow, ncol, advance_by='col', fig_kw={}, gridspec_kw={},
                 suptitle=None, suptitle_kw={}):
        """Return a FigGroup object to create subplots over multiple figures.

        Parameters
        ----------
        nrow, ncol : int
            Number of rows and columns in each figure.
        advance_by : {'col', 'row'}, optional
            Advance to next plot by incrementing over columns and then rows
            ('col'), or by incrementing over rows and then columns ('row').
        fig_kw : dict, optional
            Dict of general figure inputs to the plt.subplots() function call
            (e.g. figsize).
        gridspec_kw : dict, optional
            Dict of inputs to the plt.subplots() function call to
            specify the subplot grid (e.g. left, right, top, bottom, wspace,
            hspace, width_ratios, height ratios).
        suptitle : str, optional
            Supertitle for figures.
        suptitle_kw : dict, optional
            Dict of inputs to the plt.suptitle() function call.

        Returns
        -------
        self : FigGroup object
            The FigGroup object has all the the inputs to __init__ as
            data attributes, as well as the following data attributes:
              fig_list : list of plt.figure objects
                Figures in the group.
              axes_list : list of arrays of plt.axes objects
                Array of axes for each subplot in each figure.
              ifig, row, col : int
                Index of current figure and subplot row and column.
              fig, axes : plt.figure and plt.axes array
                Handles for current figure.
              ax : plt.axes object
                Axes of current subplot.

            And it has the following methods:
              newfig() : Create a new figure.
              next() : Advance to the next subplot.
              scf() : Set the current figure to the specified figure index.
              subplot() : Set the subplot to the specified row and column.
        """
        self.nrow, self.ncol, self.advance_by = nrow, ncol, advance_by
        self.fig_kw, self.gridspec_kw = fig_kw, gridspec_kw
        self.suptitle, self.suptitle_kw = suptitle, suptitle_kw
        #self.sharex, self.sharey = fig_kw.get('sharex'), fig_kw.get('sharey')
        self.fig_list, self.axes_list = [], []
        self.ifig, self.row, self.col = -1, -1, -1
        self.fig, self.axes, self.ax = None, None, None

    def __repr__(self):
        s = 'Figure %d, row %d, col %d' % (self.ifig, self.row, self.col)
        return s

    def newfig(self):
        """Create a new figure with the specified subplots.
        """
        self.ifig += 1
        self.row, self.col, self.ax = -1, -1, None
        fig, axes = plt.subplots(self.nrow, self.ncol,
                                 gridspec_kw=self.gridspec_kw, **self.fig_kw)
        if self.suptitle is not None:
            fig.suptitle(self.suptitle, **self.suptitle_kw)

        # Make sure axes array has consistent shape if nrow == 1 or ncol == 1
        axes = axes.reshape((self.nrow, self.ncol))

        self.fig, self.axes = fig, axes
        self.fig_list.append(fig)
        self.axes_list.append(axes)
        return None

    def subplot(self, row, col):
        """ Set the subplot axes to the specified row and column."""
        self.row, self.col = row, col
        self.ax = self.axes[row, col]
        plt.sca(self.ax)
        return None

    def scf(self, i, row=0, col=0):
        """Set the current figure to the specified figure index."""
        self.fig = self.fig_list[i]
        self.axes = self.axes_list[i]
        self.subplot(row, col)
        return None

    def next(self):
        """Advance to the next subplot."""
        row, col, nrow, ncol = self.row, self.col, self.nrow, self.ncol
        if self.advance_by == 'col':
            row = max(row, 0)
            col += 1
            if col == ncol:
                col = 0
                row += 1
        else:
            col = max(col, 0)
            row += 1
            if row == nrow:
                row = 0
                col += 1
        if row == nrow or col == ncol:
            row, col = 0, 0
        if row == 0 and col == 0:
            self.newfig()
        self.subplot(row, col)
        return None


# ----------------------------------------------------------------------
def savefigs(namestr, ext='eps', fignums=None, merge=False, **kwargs):
    """Save list of figures to numbered files with same naming convention.

    Figures are saved to files named namestr`dd`.ext where `dd` is the
    figure number.

    Parameters
    ----------
    namestr : str
        String for numbered file name.
    ext : str, optional
        File extension.
    fignums : list of ints, optional
        List of figures to save. If omitted, then all open figures are
        saved.
    merge : bool, optional
        If True, merge PDFs into a single file and delete the individual
        figure PDFs.  Only applicable if ext is 'pdf'.
    **kwargs : keyword arguments for plt.savefig()
    """

    if fignums is None:
        fignums = plt.get_fignums()
    filenames = []
    for n in fignums:
        filn = '%s%02d.%s' % (namestr, n, ext)
        filenames.append(filn)
        print('Saving to ' + filn)
        fig = plt.figure(n)
        fig.savefig(filn, **kwargs)

    if merge:
        if ext.lower() != 'pdf':
            raise ValueError('Option merge only available for pdf')
        else:
            outfile = namestr + '.pdf'
            print('Merging to ' + outfile)
            pdfmerge(filenames, outfile, delete_indiv=True)


# ----------------------------------------------------------------------
def symm_colors(plotdata):
    """Return True if data has both positive & negative values."""
    if plotdata.min() * plotdata.max() > 0:
        symmetric = False
    else:
        symmetric = True
    return symmetric



# ======================================================================
# ORDERED DICTIONARIES
# ======================================================================

# ----------------------------------------------------------------------
def print_odict(od, indent=2, width=None):
    """Pretty print the contents of an ordered dictionary."""

    if width is None:
        defwidth = 20
        widths = [len(key) for key in od]
        if len(widths) == 0:
            width = defwidth
        else:
            width = max(widths) + indent + 1

    for key in od:
        s = ' ' * indent + key
        print(s.ljust(width) + str(od[key]))



# ----------------------------------------------------------------------
def odict_insert(odict, newkey, newval, pos=0):
    """Return an OrderedDict with key:value inserted at specified position.

    Parameters
    ----------
    odict : collections.OrderedDict
        Ordered dictionary to copy and insert new value into.
    newkey : string
        New key to insert.
    newval : any
        New value to insert.
    pos : int, optional
        Position to insert.  Default 0 (prepend to start of dict).

    Returns
    -------
    odict_new : collections.OrderedDict
        A copy of the dictionary with the new key:value pair inserted.
    """

    odict_new = collections.OrderedDict()
    for i, key in enumerate(odict):
        if i == pos:
            odict_new[newkey] = newval
        odict_new[key] = odict[key]
    return odict_new


# ----------------------------------------------------------------------
def odict_delete(odict, key):
    """Return an OrderedDict with selected key:value pair removed.

    Parameters
    ----------
    odict : collections.OrderedDict
        Ordered dictionary to copy and insert new value into.
    key : string
        Key to delete.

    Returns
    -------
    odict_new : collections.OrderedDict
        A copy of the dictionary the new key:value pair deleted.
    """

    odict_new = collections.OrderedDict()
    for k in odict.keys():
        if k != key:
            odict_new[k] = odict[k]
    return odict_new


# ======================================================================
# LISTS / 1D NUMPY ARRAYS
# ======================================================================

# ----------------------------------------------------------------------
def makelist(input):
    """Return a list/array object from the input.

    If input is a single value (e.g. int, str, etc.) then output
    is a list of length 1.  If input is already a list or array, then
    output is the same as input.
    """
    if isinstance(input, list) or isinstance(input, np.ndarray):
        output = input
    else:
        output = [input]
    return output


# ----------------------------------------------------------------------
def strictly_increasing(L):
    """Return True if list L is strictly increasing."""
    return all(x < y for x, y in zip(L, L[1:]))


# ----------------------------------------------------------------------
def strictly_decreasing(L):
    """Return True if list L is strictly decreasing."""
    return all(x > y for x, y in zip(L, L[1:]))


# ----------------------------------------------------------------------
def non_increasing(L):
    """Return True if list L is non-increasing."""
    return all(x >= y for x, y in zip(L, L[1:]))


# ----------------------------------------------------------------------
def non_decreasing(L):
    """Return True if list L is non-decreasing."""
    return all(x <= y for x, y in zip(L, L[1:]))


# ----------------------------------------------------------------------
def find_closest(arr, val):
    """Return the closest value to val and its index in a 1D array.

    If the closest value occurs more than once in the array, the index
    of the first occurrence is returned.

    Usage: closest_val, ind = find_closest(arr, val)

    """
    diff = abs(arr-val)
    ind = int(diff.argmin())
    closest_val = float(arr[ind])
    return closest_val, ind


# ======================================================================
# DATE/TIME
# ======================================================================

# ----------------------------------------------------------------------
def disptime(fmt=None):
    now = datetime.now()

    if fmt == None:
        fmt = '%02d/%02d/%02d %02d:%02d:%02d'
    now = (fmt % (now.month, now.day, now.year, now.hour, now.minute,
           now.second))
    print(now)


# ----------------------------------------------------------------------
def timedelta_convert(dt, units='s'):
    """Return an np.timedelta64 time in the selected units.

    Valid units: 'ns', 'ms', 's', 'm', 'h', 'D'
    """
    return dt / np.timedelta64(1, units)


# ======================================================================
# MONTHS AND SEASONS
# ======================================================================

# ----------------------------------------------------------------------
def isleap(year):
    """Return True if year is a leap year, False otherwise."""
    return year % 4 == 0


# ----------------------------------------------------------------------
def month_str(month, upper=True):
    """Returns the string e.g. 'JAN' corresponding to month"""

    months=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
            'sep', 'oct', 'nov', 'dec']

    mstr = months[month - 1]
    if upper:
        mstr = mstr.upper()
    return mstr


# ----------------------------------------------------------------------
def days_per_month(leap=False):
    """Return array with number of days per month."""

    ndays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leap:
        ndays[1]+= 1
    return ndays


# ----------------------------------------------------------------------
def days_this_month(year, month):
    """Return the number of days in a selected month and year.

    Both inputs must be integers, and month is the numeric month 1-12.
    """
    ndays = days_per_month(isleap(year))
    return ndays[month - 1]


# ----------------------------------------------------------------------
def season_months(season):
    """
    Return list of months (1-12) for the selected season.

    Valid input seasons are:
    ssn=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
         'sep', 'oct', 'nov', 'dec', 'djf', 'mam', 'jja', 'son',
         'mayjun', 'julaug', 'marapr', 'jjas', 'ond', 'ann']
    """

    ssn=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
         'sep', 'oct', 'nov', 'dec', 'djf', 'mam', 'jja', 'son',
         'mayjun', 'julaug', 'marapr', 'jjas', 'ond', 'ann']

    imon = [1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, [1,2,12], [3,4,5], [6,7,8], [9,10,11],
            [5,6], [7,8], [3,4], [6,7,8,9], [10,11,12], range(1,13)]

    try:
        ifind = ssn.index(season.lower())
    except ValueError:
        raise ValueError('Season not found! Valid seasons: ' + ', '.join(ssn))

    months = imon[ifind]

    # Make sure the output is a list
    if isinstance(months, int):
        months =[months]

    return months


# ----------------------------------------------------------------------
def season_days(season, leap=False):
    """
    Returns indices (1-365 or 1-366) of days of the year for the input season.

    Valid input seasons are as defined in the function season_months().
    """

    # Index of first day of each month
    ndays = days_per_month(leap=leap)
    ndays.insert(0,1)
    days = np.cumsum(ndays)

    # Index of months for this season
    imon = season_months(season)

    # Days of the year for this season
    if isinstance(imon, list):
        # Iterate over months in this season
        idays=[]
        for m in imon:
            idays += range(days[m-1], days[m])
    else:
        # Single month
        idays = range(days[imon-1], days[imon])

    return idays


# ----------------------------------------------------------------------
def jday_to_mmdd(jday, year=None):
    """
    Returns numeric month and day for day of year (1-365 or 1-366).

    If year is None, a non-leap year is assumed.
    Usage: mon, day = jday_to_mmdd(jday, year)
    """
    if year is None or not isleap(year):
        leap = False
    else:
        leap = True

    ndays = days_per_month(leap)
    iday = np.cumsum(np.array([1] + ndays))
    if jday >= iday[-1]:
        raise ValueError('Invalid input day %d' + str(jday))

    BIG = 1000 # Arbitrary big number above 366
    d = np.where(jday >= iday, jday - iday + 1, BIG)
    ind = d.argmin()
    mon = ind + 1
    day = d[ind]

    return mon, day


# ----------------------------------------------------------------------
def mmdd_to_jday(month, day, year=None):
    """
    Returns Julian day of year (1-365 or 1-366) for day of month.

    If year is None, a non-leap year is assumed.
    Usage: mon, day = jday_to_mmdd(jday, year)
    """
    if year is None or not isleap(year):
        leap = False
    else:
        leap = True

    days = season_days('ann', leap)
    mmdd = {}
    for mm in range(1, 13):
        mmdd[mm] = {}
    for d in days:
        mm, dd = jday_to_mmdd(d, year)
        mmdd[mm][dd] = d
    jday = mmdd[month][day]

    return jday

# ----------------------------------------------------------------------
def pentad_to_jday(pentad, pmin=0, day=3):
    """
    Returns day of year for a pentad (indexed from pmin).

    Input day determines which day (1-5) in the pentad to return.
    Usage: jday = pentad_to_jday(pentad, pmin)
    """

    if day not in range(1, 6):
        raise ValueError('Invalid day ' + str(day))

    jday = 5*(pentad - pmin) + day
    return jday
