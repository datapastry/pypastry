import json
from glob import glob

from pypastry.core import DISPLAY_PATH
from git import Repo
from pandas import DataFrame


def cache_display():
    print("Regenerating cache")

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
    recent_results = results[-5:]
    results_dataframe = DataFrame(recent_results)
    display = repr(results_dataframe)
    with open(DISPLAY_PATH, 'w') as output_file:
        output_file.write(display)


RESULTS_GLOB = "results/*.json"