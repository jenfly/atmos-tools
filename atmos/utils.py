'''
Some general purpose utility functions used by other modules in this package.
'''

import numpy as np

# ----------------------------------------------------------------------
def print_if(msg, condition, printfunc=False):
    ''' Print msg if condition is True'''
    if condition:
        if printfunc:
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
# MONTHS AND SEASONS
# ----------------------------------------------------------------------

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
        print('ERROR: Season not found! Valid seasons are:')
        print(ssn)
        raise

    return imon[ifind]

# ----------------------------------------------------------------------
def season_days(season):
    '''
    Returns indices (1-365) of days of the year for the selected season.

    Valid input seasons are as defined in the function season_months().
    Days are for a non-leap year.
    '''

    # Index of first day of each month
    days=np.cumsum([1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

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
