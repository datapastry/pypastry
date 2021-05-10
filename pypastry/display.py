"""
Handle displaying results.

This code needs to display results really fast. That's why it has some odd things:
 - Inline imports
 - Lack of typing for the results_repo to avoid unnecessary imports

This is because I don't like having to wait a second for things to be imported.
I want my pastry now!
"""
import os
from typing import Any, Dict, List, Iterator, TYPE_CHECKING

from pandas import Series

from pypastry.paths import DISPLAY_PATH, DISPLAY_DIR
if TYPE_CHECKING:
    import pypastry


def cache_display(results_from_repo: Iterator['pypastry.experiment.results.Result']) -> None:
    results_dataframe = _get_results_dataframe(results_from_repo)
    display = repr(results_dataframe)

    try:
        os.mkdir(DISPLAY_DIR)
    except FileExistsError:
        pass

    with open(DISPLAY_PATH, 'w') as output_file:
        output_file.write(display)


def _get_results_dataframe(results_from_repo: Iterator['pypastry.experiment.results.Result']) -> 'DataFrame':
    from pandas import DataFrame, set_option
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)
    set_option('display.max_colwidth', -1)
    results = []
    for repo_result in results_from_repo:
        data = repo_result.data
        result = {
            'Git hash': data["git_hash"] if "git_hash" in data else "Unavailable",
            'Git summary': data["git_summary"] if "git_summary" in data else "Unavailable",
            'Dataset hash': data['dataset']['hash'][:8],
            'Run start': data['run_start'][:19],
            'Model': data['model_info']['type'],
            'Duration (s)': "{:.2f}".format(data['run_seconds']),
        }

        try:
            scores = DataFrame(data['results'])
            for row in scores.itertuples():
                result[row.Index] = "{:.3f} ± {:.3f}".format(row.test_score, row.test_score_sem)
        except ValueError:
            result['Score'] = "{:.3f} ± {:.3f}".format(data['results']['test_score'],
                                                       data['results']['test_score_sem'])

        results.append(result)
    results_dataframe = DataFrame(results)
    return results_dataframe.sort_values(by='Run start').reset_index()


def print_cache_file(limit=False):
    with open(DISPLAY_PATH) as display_file:
        print(display_file.read())
        # read_lines = display_file.read()
        # read_list = read_lines.split("\n")
        # if limit:
        #     limit = min(limit, len(read_list)-3)
        #     # to avoid printing more than there is
        # else:
        #     limit = len(read_list)-3
        # print(read_list[0])
        # print("\n".join(read_list[-(2+limit):-2]))
        # print(read_list[-1])

