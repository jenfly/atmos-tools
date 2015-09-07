'''
Some general purpose utility functions used by other modules in this package.
'''

import numpy as np
from datetime import datetime

# ======================================================================
# PRINTING
# ======================================================================

# ----------------------------------------------------------------------
def print_if(msg, condition, printfunc=None):
    ''' Print msg if condition is True'''
    if condition:
        if printfunc is not None:
            printfunc(msg)
        else:
            print(msg)

# ----------------------------------------------------------------------
def print_odict(od, indent=2, width=20):
    '''Pretty print the contents of an ordered dictionary.'''
    for key in od:
        s = ' ' * indent + key
        print(s.ljust(width) + str(od[key]))


# ----------------------------------------------------------------------
def disptime(fmt=None):
    now = datetime.now()

    if fmt == None:
        fmt = '%02d/%02d/%02d %02d:%02d:%02d'
    now = (fmt % (now.month, now.day, now.year, now.hour, now.minute,
           now.second))
    print(now)


# ======================================================================
# INCREASING / DECREASING LISTS
# ======================================================================

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


# ======================================================================
# MONTHS AND SEASONS
# ======================================================================

# ----------------------------------------------------------------------
def month_str(month, upper=True):
    '''Returns the string e.g. 'JAN' corresponding to month'''

    months=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
            'sep', 'oct', 'nov', 'dec']

    mstr = months[month - 1]
    if upper:
        mstr = mstr.upper()
    return mstr

# ----------------------------------------------------------------------
def days_per_month(leap=False):
    '''Returns array with number of days per month.'''

    ndays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if leap:
        ndays[1]+= 1
    return ndays

# ----------------------------------------------------------------------
def season_months(season):
    '''
    Returns list of months (1-12) for the selected season.

    Valid input seasons are:
    ssn=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',
         'sep', 'oct', 'nov', 'dec', 'djf', 'mam', 'jja', 'son',
         'mayjun', 'julaug', 'marapr', 'jjas', 'ond', 'ann']
    '''

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

    return imon[ifind]

# ----------------------------------------------------------------------
def season_days(season, leap=False):
    '''
    Returns indices (1-365 or 1-366) of days of the year for the input season.

    Valid input seasons are as defined in the function season_months().
    '''

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
