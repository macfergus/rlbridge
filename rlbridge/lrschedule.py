import copy
from collections import namedtuple


Breakpoint = namedtuple('Breakpoint', ['endpoint', 'lr'])


class LRSchedule:
    def __init__(self, changes, final):
        self._changes = copy.copy(changes)
        self._final = final

    def lookup(self, position):
        for breakpoint in self._changes:
            if position < breakpoint.endpoint:
                return breakpoint.lr
        return self._final

    @classmethod
    def fixed(cls, lr):
        return cls([], lr)

    @classmethod
    def from_dicts(cls, dictlist):
        final = None
        breakpoints = []
        for d in dictlist:
            if 'finally' in d:
                final = d['finally']
            else:
                breakpoints.append(
                    Breakpoint(endpoint=d['until'], lr=float(d['lr']))
                )
        assert final is not None
        breakpoints.sort()
        return LRSchedule(breakpoints, final)
