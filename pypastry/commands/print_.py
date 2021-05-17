import argparse

from pypastry.display import print_cache_file, cache_display, _get_results_dataframe
from pypastry.paths import RESULTS_PATH
from pypastry.experiment.results import ResultsRepo


def run():
    parser = argparse.ArgumentParser(prog='pastry print')
    parser.add_argument('-l', '--limit', type=int, default=None, help='Limit lines to print')
    parser.add_argument('-e', '--export', type=str, required=False, help='File to output the results in CSV format')

    args = parser.parse_args(sys.argv[2:])
    if args.export is not None:
        results = get_results()
        results_dataframe = _get_results_dataframe(results)
        results_dataframe.to_csv(args.export)
        return
    try:
        print_cache_file(args.limit)
    except FileNotFoundError:
        results = get_results()
        cache_display(results)
        print_cache_file(args.limit)


def get_results():
    results_repo = ResultsRepo(RESULTS_PATH)
    results = results_repo.get_results()
    return results
