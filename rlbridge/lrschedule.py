import copy
from collections import namedtuple

Breakpoint = namedtuple('Breakpoint', ['endpoint', 'lr'])


class LRSchedule:
    def __init__(self, changes, final):
        self._changes = copy.copy(changes)
        self._final = final

    def lookup(self, position):
        for breakpt in self._changes:
            if position < breakpt.endpoint:
                return breakpt.lr
        return self._final

    @classmethod
    def fixed(cls, lr):
        return cls([], lr)

    @classmethod
    def from_dicts(cls, dictlist):
        final = None
        breakpoints = []
        for dct in dictlist:
            if 'finally' in dct:
                final = dct['finally']
            else:
                breakpoints.append(
                    Breakpoint(endpoint=dct['until'], lr=float(dct['lr']))
                )
        assert final is not None
        breakpoints.sort()
        return LRSchedule(breakpoints, final)
