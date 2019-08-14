from pypastry.display import print_cache_file, cache_display
from pypastry.paths import RESULTS_PATH


def run():
    try:
        print_cache_file()
    except FileNotFoundError:
        from pypastry.experiment.results import ResultsRepo
        from git import Repo
        results_repo = ResultsRepo(RESULTS_PATH)
        results = results_repo.get_results(Repo('.'))
        cache_display(results)
        print_cache_file()
