__all__ = [
    'parse_options',
]


def parse_options(options_str):
    pairs = options_str.split(',')
    options = {}
    for pair in pairs:
        k, v = pair.split('=')
        options[k] = v
    return options
