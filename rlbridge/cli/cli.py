import argparse
import sys

from . import (benchmark, demogame, diagnose, evaluate, initbot, pretrain,
               prune, rename, selfplay, stats)


def cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    commands = [
        benchmark.Benchmark(),
        demogame.DemoGame(),
        diagnose.Diagnose(),
        evaluate.Evaluate(),
        initbot.InitBot(),
        pretrain.Pretrain(),
        prune.Prune(),
        rename.Rename(),
        selfplay.SelfPlay(),
        stats.Stats(),
    ]
    command_map = {}
    for command in commands:
        subparser = subparsers.add_parser(
            command.name(),
            help=command.description())
        command.register_arguments(subparser)
        command_map[command.name()] = command

    args = parser.parse_args()
    if not args.command:
        parser.print_usage()
        sys.exit(0)
    command_map[args.command].run(args)
