import importlib

from ..io import open_h5file_if_necessary

__all__ = [
    'init_bot',
    'save_bot',
    'load_bot',
]


def load_bot_module(bot_type):
    mod_path = 'rlbridge.bots.' + bot_type
    return importlib.import_module(mod_path)


def init_bot(bot_type, options, metadata):
    m = load_bot_module(bot_type)
    init_fn = getattr(m, 'init')
    return init_fn(options, metadata)


def save_bot(bot, outputfile):
    bot_type = bot.bot_type()
    m = load_bot_module(bot_type)
    save_fn = getattr(m, 'save')
    with open_h5file_if_necessary(outputfile, 'w') as outf:
        outf.attrs['bot_type'] = bot_type
        metadata = outf.create_group('metadata')
        for k, v in bot.metadata.items():
            metadata.attrs[k] = v
        bot_data = outf.create_group('bot_data')
        save_fn(bot, bot_data)


def load_bot(inputfile):
    with open_h5file_if_necessary(inputfile, 'r') as inf:
        bot_type = inf.attrs['bot_type']
        m = load_bot_module(bot_type)
        load_fn = getattr(m, 'load')
        metadata_group = inf['metadata']
        metadata = {}
        for k in metadata_group.attrs:
            metadata[k] = metadata_group.attrs[k]
        bot_data = inf['bot_data']
        return load_fn(bot_data, metadata)
