import copy
from collections import namedtuple

Breakpoint = namedtuple('Breakpoint', ['endpoint', 'value'])


class Schedule:
    def __init__(self, changes, final):
        self._changes = copy.copy(changes)
        self._final = final

    def lookup(self, position):
        for breakpt in self._changes:
            if position < breakpt.endpoint:
                return breakpt.value
        return self._final

    @classmethod
    def fixed(cls, value):
        return cls([], value)

    @classmethod
    def from_dicts(cls, dictlist):
        final = None
        breakpoints = []
        for dct in dictlist:
            if 'finally' in dct:
                final = dct['finally']
            else:
                breakpoints.append(
                    Breakpoint(endpoint=dct['until'], value=float(dct['value']))
                )
        assert final is not None
        breakpoints.sort()
        return Schedule(breakpoints, final)
