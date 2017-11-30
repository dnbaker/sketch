from _hll import *

def addh(hll, item):
    ''' Hash an item and add it to the hll sketch. '''
    if item.__hash__:
        hll.add(item.__hash__(item))
    else:
        hll.addh_(int(item))

hll.addh = addh
__doc__ = "HyperLogLog module"

__all__ = [hll]
