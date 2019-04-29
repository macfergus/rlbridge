from contextlib import contextmanager

import h5py

__all__ = [
    'open_h5file_if_necessary',
]


@contextmanager
def open_h5file_if_necessary(filename_or_h5file, mode='r'):
    if isinstance(filename_or_h5file, str):
        with h5py.File(filename_or_h5file, mode) as f:
            yield f
    else:
        yield filename_or_h5file
