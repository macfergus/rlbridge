import json
import os

from .bots import load_bot, save_bot


class UninitializedError(Exception):
    pass


class Workspace:
    def __init__(self, basedir):
        self.base_dir = basedir
        if not os.path.exists(self.base_dir):
            raise UninitializedError()

        self.bot_dir = os.path.join(basedir, 'bots')
        self.eval_dir = os.path.join(basedir, 'eval')
        self.state_file = os.path.join(basedir, 'state')
        self.eval_db_file = os.path.join(basedir, 'evaluation.db')

    def store_bot(self, bot):
        bot_path = os.path.join(self.bot_dir, bot.identify())
        save_bot(bot, bot_path)
        return bot_path

    def store_bot_for_eval(self, bot):
        bot_path = os.path.join(self.eval_dir, bot.identify())
        save_bot(bot, bot_path)
        return bot_path


def init_workspace(run_id, start_from):
    home_dir = os.path.expanduser("~")
    root_dir = os.path.join(home_dir, '.rlbridge', 'work')
    workspace_dir = os.path.join(root_dir, run_id)

    os.makedirs(workspace_dir)

    bot_dir = os.path.join(workspace_dir, 'bots')
    os.makedirs(bot_dir)
    eval_dir = os.path.join(workspace_dir, 'eval')
    os.makedirs(eval_dir)

    workspace = Workspace(workspace_dir)

    start_bot = load_bot(start_from)
    path = workspace.store_bot(start_bot)
    workspace.store_bot_for_eval(start_bot)
    state_path = os.path.join(workspace_dir, 'state')
    with open(state_path, 'w') as outf:
        outf.write(json.dumps({
            'learn': path,
            'ref': [path],
        }))

    return workspace


def open_workspace(run_id):
    home_dir = os.path.expanduser("~")
    root_dir = os.path.join(home_dir, '.rlbridge', 'work')
    workspace_dir = os.path.join(root_dir, run_id)

    return Workspace(workspace_dir)
