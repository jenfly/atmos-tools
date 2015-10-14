"""
Some general purpose utility functions used by other modules in this package.
"""

from __future__ import division
import numpy as np
import collections
from datetime import datetime

# ======================================================================
# PRINTING
# ======================================================================

# ----------------------------------------------------------------------
def print_if(msg, condition, printfunc=None):
    """ Print msg if condition is True"""
    if condition:
        if printfunc is not None:
            printfunc(msg)
        else:
            print(msg)


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

