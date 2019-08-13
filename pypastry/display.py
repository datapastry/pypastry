"""
Handle displaying results.

This code needs to display results really fast. That's why it has some odd things:
 - Inline imports
 - Lack of typing for the results_repo to avoid unnecessary imports

This is because I don't like having to wait a second for things to be imported.
I want my pastry now!
"""

from pypastry.paths import DISPLAY_PATH, RESULTS_PATH, REPO_PATH


class ResultsDisplay:
    def __init__(self, results_repo):
        self.results_repo = results_repo

    def cache_display(self):
        from pandas import DataFrame
        print("Regenerating cache")

        results = []
        for result in self.results_repo.get_results():
            data = result.data
            result = {
                'Git hash': result.git_hash,
                'Dataset hash': data['dataset']['hash'][:8],
                'Run start': data['run_start'][:19],
                'Model': data['model_info']['type'],
                'Score': "{:.3f} ± {:.3f}".format(data['results']['test_score'],
                                                  data['results']['test_score_sem']),
                'Duration (s)': "{:.2f}".format(data['run_seconds']),
            }
            results.append(result)
        results.sort(key=lambda row: row['Run start'])
        recent_results = results[-5:]
        results_dataframe = DataFrame(recent_results)
        display = repr(results_dataframe)

        with open(DISPLAY_PATH, 'w') as output_file:
            output_file.write(display)


def print_display():
    try:
        _print_cache_file()
    except FileNotFoundError:
        from pypastry.experiment.results import ResultsRepo
        from git import Repo
        display = ResultsDisplay(ResultsRepo(RESULTS_PATH, Repo(REPO_PATH)))
        display.cache_display()
        _print_cache_file()


def _print_cache_file():
    with open(DISPLAY_PATH) as display_file:
        print(display_file.read())
