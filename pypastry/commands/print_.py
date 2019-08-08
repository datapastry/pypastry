import json
from glob import glob

from git import Repo
from pandas import DataFrame
from pypastry import RESULTS_GLOB


def print_results():
    results = []

    repo = Repo('.')

    for path in glob(RESULTS_GLOB):
        with open(path) as results_file:

            git_hash = next(repo.iter_commits(paths=path)).hexsha[:8]
            result_json = json.load(results_file)
            result = {
                'Git hash': git_hash,
                'Dataset hash': result_json['dataset']['hash'][:8],
                'Run start': result_json['run_start'][:19],
                'Model': result_json['model_info']['type'],
                'Score': "{:.3f} ± {:.3f}".format(result_json['results']['test_score'],
                                            result_json['results']['test_score_sem']),
                'Duration (s)': "{:.2f}".format(result_json['run_seconds']),
            }
            results.append(result)

    results.sort(key=lambda row: row['Run start'])
    results_dataframe = DataFrame(results)
    print(results_dataframe)