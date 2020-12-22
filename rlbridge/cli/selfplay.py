import json
import os
import shutil
import time

import yaml

from .. import kerasutil
from ..bots import load_bot, save_bot
from ..mputil import MPLogManager
from ..selfplay import SelfPlayManager
from ..workspace import UninitializedError, init_workspace, open_workspace
from .command import Command


class SelfPlay(Command):
    def register_arguments(self, parser):
        parser.add_argument('--start-from')
        parser.add_argument('--config', '-c', required=True)
        parser.add_argument(
            '--evaluate-only', default=False, action='store_true'
        )

    def run(self, args):
        kerasutil.set_tf_options(disable_gpu=True)
        conf = yaml.safe_load(open(args.config))

        if args.start_from:
            workspace = init_workspace(conf['run_id'], args.start_from)
        else:
            try:
                workspace = open_workspace(conf['run_id'])
            except UninitializedError:
                print(f'Run {conf["run_id"]} is not initialized.')
                print('Run with --start-from to initialize.')
                return

        log_mgr = MPLogManager()
        log_mgr.start()
        logger = log_mgr.get_logger()

        selfplay = SelfPlayManager(
            workspace=workspace,
            config=conf,
            logger=logger,
            evaluate_only=args.evaluate_only
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
