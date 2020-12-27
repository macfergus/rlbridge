import copy
import json

from .. import bots
from ..io import parse_options
from .command import Command


class InitBot(Command):
    def register_arguments(self, parser):
        parser.add_argument('--name')
        parser.add_argument('--metadata')
        parser.add_argument('--options')
        parser.add_argument('bot_type')
        parser.add_argument('output_file')

    def run(self, args):
        if args.options:
            options = parse_options(args.options)
        else:
            options = {}

        metadata = {}
        if args.name:
            metadata['name'] = args.name
        else:
            metadata['name'] = args.bot_type
        for k, v in options.items():
            metadata[f'model_option__{k}'] = v
        if args.metadata:
            metadata.update(json.loads(args.metadata))

        bot = bots.init_bot(args.bot_type, options, metadata)
        print(bot.model.summary())
        bots.save_bot(bot, args.output_file)
