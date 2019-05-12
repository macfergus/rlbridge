from ... import kerasutil
from . import encoder, model
from .bot import LSTMBot
from .losses import policy_loss


def init(options, metadata):
    enc = encoder.Encoder()
    mod = model.construct_model(
        input_shape=enc.input_shape(),
        lstm_size=int(options.get('lstm_size', 512)),
        lstm_depth=int(options.get('lstm_depth', 2))
    )
    return LSTMBot(mod, metadata)


def save(bot, h5group):
    model_group = h5group.create_group('model')
    kerasutil.save_model_to_hdf5_group(bot.model, model_group)


def load(h5group, metadata):
    model_group = h5group['model']
    mod = kerasutil.load_model_from_hdf5_group(
        model_group,
        custom_objects={'policy_loss': policy_loss}
    )
    return LSTMBot(mod, metadata)
