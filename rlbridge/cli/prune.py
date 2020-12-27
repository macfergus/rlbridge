import json
import os

from ..workspace import open_workspace
from .command import Command


class Prune(Command):
    def register_arguments(self, parser):
        parser.add_argument('run_id')

    def run(self, args):
        workspace = open_workspace(args.run_id)
        status = json.load(open(workspace.state_file))
        known_bots = set(status['ref']) | set([status['learn']])
        num_pruned = 0
        for bot_file in os.listdir(workspace.bot_dir):
            full_path = os.path.join(workspace.bot_dir, bot_file)
            if not os.path.isfile(full_path):
                continue
            keep = False
            for known_bot in known_bots:
                if os.path.samefile(full_path, known_bot):
                    keep = True
            if keep:
                print('keep', full_path)
            else:
                print('remove', full_path)
                os.remove(full_path)
                num_pruned += 1
        print(f'Pruned {num_pruned} bots')
