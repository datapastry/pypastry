import argparse
import sys

from pypastry.evaluation import run_experiment


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--force', action='store_true', help="Force a re-run of the experiment")

    args = parser.parse_args(sys.argv[2:])

    sys.path.append('.')
    import pie
    experiment = pie.get_experiment()

    run_experiment(experiment, args.force)
