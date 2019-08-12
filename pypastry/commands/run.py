import argparse
import sys

from pypastry.experiment.evaluation import run_experiment


def run():
    parser = argparse.ArgumentParser(prog='pastry run')
    parser.add_argument('-f', '--force', action='store_true', help='Force a re-run of the experiment')
    parser.add_argument('-m', '--message', type=str, required=True, help='Git commit message')

    args = parser.parse_args(sys.argv[2:])

    sys.path.append('.')
    import pie
    experiment = pie.get_experiment()

    run_experiment(experiment, args.force, args.message)
