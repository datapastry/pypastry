import argparse
import sys

from pypastry.experiment.evaluation import run_experiment


def run():
    parser = argparse.ArgumentParser(prog='pastry run')
    parser.add_argument('-m', '--message', default="", type=str, help='Summary message about the experiment.')
    parser.add_argument('-f', '--force', action='store_true', help='Force a re-run of the experiment')
    parser.add_argument('-p', '--no-print', action='store_true', help='Do not print results.')

    args = parser.parse_args(sys.argv[2:])

    sys.path.append('.')
    import pie
    experiment = pie.get_experiment()
    force = args.force
    message = args.message

    run_experiment(experiment, message, force, show_results=not args.no_print)
