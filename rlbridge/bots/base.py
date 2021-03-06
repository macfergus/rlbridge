import copy


class UnrecognizedOptionError(Exception):
    pass


class Bot:
    """Base class for bridge bots."""
    def __init__(self, metadata):
        self.metadata = copy.deepcopy(metadata)

    def bot_type(self):
        module_name = self.__class__.__module__
        assert module_name.startswith('rlbridge.bots.')
        return module_name.split('.')[2]

    def name(self):
        return self.metadata['name']

    def identify(self):
        return self.name()

    def set_option(self, key, value):
        raise UnrecognizedOptionError(key)

    def get_diagnostics(self):
        """Return a dictionary explaining the last decision."""
        return {}
