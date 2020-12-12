import json
import os
import time

import yaml

from .command import Command
from ..mputil import MPLogManager
from ..selfplay import SelfPlayManager


class SelfPlay(Command):
    def register_arguments(self, parser):
        parser.add_argument('--start-from')
        parser.add_argument('--config', '-c', required=True)
        parser.add_argument('work_dir')

    def run(self, args):
        bot_state_path = os.path.join(args.work_dir, 'bot_status')
        if args.start_from:
            with open(bot_state_path, 'w') as outf:
                outf.write(json.dumps({
                    'learn': args.start_from,
                    'ref': [args.start_from],
                }))

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
                if not selfplay.is_healthy():
                    logger.log('Self-play workers crashed; shutting down')
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            logger.log('received ^C, shutting down...')
        finally:
            selfplay.stop()
            log_mgr.stop()
