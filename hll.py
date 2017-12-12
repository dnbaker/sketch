from _hll import _hll


class hll(_hll):
    def addh(self, item):
        ''' Hash an item and add it to the hll sketch. '''
        if item.__hash__:
            self.add(item.__hash__())
        else:
            self.addh_(int(item))


__doc__ = "HyperLogLog module"
__all__ = [hll]
