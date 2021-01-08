from ... import kerasutil
from . import encoder, encoder2d, model
from .bot import ConvBot
from .losses import policy_loss


def init(options, metadata):
    structure = options.get('conv', '1d')
    if structure == '1d':
        enc = encoder.Encoder()
    else:
        enc = encoder2d.Encoder2D()
    mod = model.construct_model(
        input_shape=enc.input_shape(),
        structure=structure,
        num_filters=int(options.get('num_filters', 128)),
        kernel_size=int(options.get('kernel_size', 21)),
        num_layers=int(options.get('num_layers', 5)),
        state_size=int(options.get('state_size', 64)),
        flat_channels=int(options.get('flat_channels', 4)),
        hidden_size=int(options.get('hidden_size', 64)),
        regularization=float(options.get('regularization', 0.0)),
        kernel_reg=float(options.get('kernel_reg', 0.0)),
        aux_outs=options.get('aux_outs')
    )
    return ConvBot(enc, mod, metadata)


def save(bot, h5group):
    model_group = h5group.create_group('model')
    encoder_group = h5group.create_group('encoder')
    if isinstance(bot.encoder, encoder.Encoder):
        structure = '1d'
    elif isinstance(bot.encoder, encoder2d.Encoder2D):
        structure = '2d'
    else:
        raise TypeError(bot.encoder)
    encoder_group.attrs['structure'] = structure
    kerasutil.save_model_to_hdf5_group(bot.model, model_group)


def load(h5group, metadata):
    model_group = h5group['model']
    if 'encoder' in h5group:
        structure = h5group['encoder'].attrs['structure']
        if structure == '1d':
            enc = encoder.Encoder()
        elif structure == '2d':
            enc = encoder2d.Encoder2D()
    else:
        enc = encoder.Encoder()
    mod = kerasutil.load_model_from_hdf5_group(
        model_group,
        custom_objects={'policy_loss': policy_loss}
    )
    return ConvBot(enc, mod, metadata)
