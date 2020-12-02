from ... import kerasutil
from . import encoder, model
from .bot import ConvBot
from .losses import policy_loss


def init(options, metadata):
    enc = encoder.Encoder()
    mod = model.construct_model(
        input_shape=enc.input_shape(),
        num_filters=int(options.get('num_filters', 128)),
        kernel_size=int(options.get('kernel_size', 21)),
        num_layers=int(options.get('num_layers', 5)),
        state_size=int(options.get('state_size', 64)),
        hidden_size=int(options.get('hidden_size', 64))
    )
    return ConvBot(mod, metadata)


def save(bot, h5group):
    model_group = h5group.create_group('model')
    kerasutil.save_model_to_hdf5_group(bot.model, model_group)


def load(h5group, metadata):
    model_group = h5group['model']
    mod = kerasutil.load_model_from_hdf5_group(
        model_group,
        custom_objects={'policy_loss': policy_loss}
    )
    return ConvBot(mod, metadata)
