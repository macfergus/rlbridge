from ..bots import load_bot, save_bot
from .command import Command


class Rename(Command):
    def register_arguments(self, parser):
        parser.add_argument('bot_in')
        parser.add_argument('new_name')

    def run(self, args):
        bot = load_bot(args.bot_in)
        bot.metadata['name'] = args.new_name
        save_bot(bot, args.bot_in)
