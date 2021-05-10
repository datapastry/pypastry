import argparse
import sys

from pypastry.experiment.evaluation import run_experiment


def run():
    parser = argparse.ArgumentParser(prog='pastry run')
    parser.add_argument('-f', '--force', action='store_true', help='Force a re-run of the experiment')

    args = parser.parse_args(sys.argv[2:])

    sys.path.append('.')
    import pie
    experiment = pie.get_experiment()
    force = args.force

    run_experiment(experiment, force)
