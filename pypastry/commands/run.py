import argparse
import sys

from git import Repo
from pypastry import display

from pypastry.experiment.evaluation import ExperimentRunner
from pypastry.experiment.results import ResultsRepo
from pypastry.paths import REPO_PATH, RESULTS_PATH


def run():
    parser = argparse.ArgumentParser(prog='pastry run')
    parser.add_argument('-f', '--force', action='store_true', help='Force a re-run of the experiment')
    parser.add_argument('-m', '--message', type=str, required=True, help='Git commit message')
    parser.add_argument('-l', '--limit', type=int, help='Limit lines to print')

    args = parser.parse_args(sys.argv[2:])

    sys.path.append('.')
    import pie
    experiment = pie.get_experiment()

    git_repo = Repo(REPO_PATH)
    results_repo = ResultsRepo(RESULTS_PATH)
    runner = ExperimentRunner(git_repo, results_repo, display)
    runner.run_experiment(experiment, args.force, args.message, args.limit)
