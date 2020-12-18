import json
import os
import shutil
import time

import yaml

from .. import kerasutil
from ..bots import load_bot, save_bot
from ..mputil import MPLogManager
from ..selfplay import SelfPlayManager
from .command import Command


class SelfPlay(Command):
    def register_arguments(self, parser):
        parser.add_argument('--start-from')
        parser.add_argument('--config', '-c', required=True)
        parser.add_argument('work_dir')

    def run(self, args):
        bot_state_path = os.path.join(args.work_dir, 'bot_status')
        if args.start_from:
            kerasutil.set_tf_options(disable_gpu=True)
            start_bot = load_bot(args.start_from)
            bot_dir = os.path.join(args.work_dir, 'bots')
            if not os.path.exists(bot_dir):
                os.mkdir(bot_dir)
            dest_path = os.path.join(bot_dir, start_bot.identify())
            save_bot(start_bot, dest_path)
            del start_bot

            with open(bot_state_path, 'w') as outf:
                outf.write(json.dumps({
                    'learn': dest_path,
                    'ref': [dest_path],
                }))
            eval_dir = os.path.join(args.work_dir, 'eval')
            if not os.path.exists(eval_dir):
                os.mkdir(eval_dir)
            shutil.copy(dest_path, eval_dir)

        if not os.path.exists(bot_state_path):
            print(f'Bot control file {bot_state_path} does not exist.')
            print('Run with --start-from to initialize.')
            return

        conf = yaml.safe_load(open(args.config))

        log_mgr = MPLogManager()
        log_mgr.start()
        logger = log_mgr.get_logger()

        selfplay = SelfPlayManager(
            state_path=bot_state_path,
            out_dir=args.work_dir,
            config=conf,
            logger=logger
        )
        selfplay.start()

        try:
            while True:
                selfplay.maintain()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.log('received ^C, shutting down...')
        finally:
            selfplay.stop()
            log_mgr.stop()
