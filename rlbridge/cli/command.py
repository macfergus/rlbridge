class Command:
    def name(self):
        return self.__class__.__name__.lower()

    def description(self):
        return None

    def register_arguments(self, parser):
        pass

    def run(self, args):
        raise NotImplementedError
