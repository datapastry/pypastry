import argparse
import sys

from pypastry.experiment.evaluation import run_experiment


def run():
    parser = argparse.ArgumentParser(prog='pastry run')
    parser.add_argument('-f', '--force', action='store_true', help='Force a re-run of the experiment')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--message', type=str, help='Git commit message')
    group.add_argument('-s', '--no-save', type=bool, action='store_true')

    args = parser.parse_args(sys.argv[2:])

    sys.path.append('.')
    import pie
    experiment = pie.get_experiment()
    force = args.force
    message = args.message
    save = not args.no_save

    run_experiment(experiment, message, force, save=save)
