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

from pypastry.paths import DISPLAY_PATH, DISPLAY_DIR
if TYPE_CHECKING:
    import pypastry


=======
def cache_display(results_from_repo: Iterator['pypastry.experiment.results.Result']) -> None:
    """
    Caches results from the machine learning experiments and selects the recent 5 results tp be displayed
    Args:
        results_from_repo (list):

    Returns:
        None

    """
    from pandas import DataFrame

    results = []
    for result in results_from_repo:
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
    recent_results = results
    results_dataframe = DataFrame(recent_results)
    display = repr(results_dataframe)

    try:
        os.mkdir(DISPLAY_DIR)
    except FileExistsError:
        pass

    with open(DISPLAY_PATH, 'w') as output_file:
        output_file.write(display)



=======
def print_cache_file(limit = False):
    """
    Prints the 5 most recent experiment results to screen
    Args:
        None

    Returns:
        None

    """
    with open(DISPLAY_PATH) as display_file:
        read_lines = display_file.read()
        read_list = read_lines.split("\n")
        if limit:
            limit = min(limit, len(read_list)-3)
        else:
            limit = len(read_list)-3
        print(read_list[0])
        print("\n".join(read_list[-(2+limit):-2]))
        print(f'Cache provides:\n{read_list[-1]}')
