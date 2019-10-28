import argparse
import sys

from pypastry.display import print_cache_file, cache_display
from pypastry.paths import RESULTS_PATH


def run():
    parser = argparse.ArgumentParser(prog='pastry print')
    parser.add_argument('-l', '--limit', type=int, default=None, help='Limit lines to print')

    args = parser.parse_args(sys.argv[2:])
    try:
        print_cache_file(args.limit)
    except FileNotFoundError:
        from pypastry.experiment.results import ResultsRepo
        from git import Repo
        results_repo = ResultsRepo(RESULTS_PATH)
        results = results_repo.get_results(Repo('.'))
        cache_display(results)
        print_cache_file(args.limit)
