import os
import tempfile

import h5py
import keras
from keras.models import load_model, save_model


def save_model_to_hdf5_group(model, outf):
    # Use Keras save_model to save the full model (including optimizer
    # state) to a file.
    # Then we can embed the contents of that HDF5 file inside ours.
    with tempfile.TemporaryDirectory() as tmpdir:
        tempfname = os.path.join(tmpdir, 'kerasmodel.h5')
        save_model(model, tempfname)
        serialized_model = h5py.File(tempfname, 'r')
        root_item = serialized_model.get('/')
        serialized_model.copy(root_item, outf, 'kerasmodel')
        serialized_model.close()


def load_model_from_hdf5_group(inf, custom_objects=None):
    # Extract the model into a temporary file. Then we can use Keras
    # load_model to read it.
    tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
    try:
        os.close(tempfd)
        serialized_model = h5py.File(tempfname, 'w')
        root_item = inf.get('kerasmodel')
        for attr_name, attr_value in root_item.attrs.items():
            serialized_model.attrs[attr_name] = attr_value
        for k in root_item.keys():
            inf.copy(root_item.get(k), serialized_model, k)
        serialized_model.close()
        return load_model(tempfname, custom_objects=custom_objects)
    finally:
        os.unlink(tempfname)


def set_tf_options(limit_memory=False):
    """Set Tensorflow options."""
    # Do the import here, not at the top, for funny forking reasons
    import tensorflow as tf
    if limit_memory:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
