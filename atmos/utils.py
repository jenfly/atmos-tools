'''
Some general purpose utility functions used by other modules in this package.
'''
# ----------------------------------------------------------------------
def print_if(msg, condition, printfunc=False):
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
