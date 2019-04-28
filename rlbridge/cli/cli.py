import argparse
import sys

from . import demogame, evaluate, initbot


def cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    commands = [
        demogame.DemoGame(),
        evaluate.Evaluate(),
        initbot.InitBot(),
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
