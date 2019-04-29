import h5py
import numpy as np

__all__ = [
    'Episode',
    'ExperienceSaver',
]


class Episode(dict):
    pass


class ExperienceSaver:
    def __init__(self, outputfile):
        self.outf = h5py.File(outputfile, 'w')

    def _create(self, name, data):
        maxshape = (None,) + data.shape[1:]
        self.outf.create_dataset(name, data=data, maxshape=maxshape)

    def _append(self, name, data):
        dataset = self.outf[name]
        assert dataset.shape[1:] == data.shape[1:]
        cur_shape = dataset.shape
        cur_n = cur_shape[0]
        new_n = data.shape[0]
        new_shape = (cur_n + new_n,) + cur_shape[1:]
        dataset.resize(new_shape)
        dataset[cur_n:cur_n + new_n] = data

    def _create_or_append(self, name, data):
        if name not in self.outf:
            self._create(name, data)
        else:
            self._append(name, data)

    def record_episode(self, episode):
        assert self.outf is not None
        for array_name in episode:
            # The data comes in with shape (timesteps, dim)
            # Reshape to shape (episode, timesteps, dim)
            data = episode[array_name]
            if not hasattr(data, 'shape'):
                data = np.array(data)
            data = data.reshape((1,) + data.shape)
            self._create_or_append(array_name, data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.outf.close()
        self.outf = None
