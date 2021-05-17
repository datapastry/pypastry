import pytest

from pypastry.display import _get_results_dataframe
from pypastry.experiment.results import Result


@pytest.fixture
def get_result_dict():

    return {
        "run_start": "2020-01-31 09:19:44.261276",
        "run_end": "2020-01-31 09:19:48.743458",
        "run_seconds": 4.482182,
        "results": {
            "test_score": {
                "mean_relative_error": 0.5,
                "mean_absolute_error": 100,
                "mean_squared_error": 1000
            },
            "test_score_sem": {
                "mean_relative_error": 0.01,
                "mean_absolute_error": 1.0,
                "mean_squared_error": 10.0
            }
        },
        "model_info": {
            "n_neighbors": 10,
            "type": "KNearestNeighbor"
        },
        "additional_info": [
            None
        ],
        "dataset": {
            "hash": "998c9dea0afb12d91a8c67f256f80b0a603dd59b",
            "columns": [
                "input",
                "output"
            ],
            "size": 100,
        },
        "git_hash": "123456781234567812345678",
        "git_summary": "12345678",
        "result_json_name": "jsonhash",
    }


def test_get_display(get_result_dict):
    result = Result(get_result_dict)
    results_dataframe = _get_results_dataframe([result])
    row = results_dataframe.iloc[0].to_dict()

    expected = {
        'Git hash': get_result_dict["git_hash"],
        'Summary': get_result_dict["git_summary"],
        'Dataset size': get_result_dict["dataset"]["size"],
        'Dataset hash': get_result_dict["dataset"]["hash"][:8],
        'Run start': get_result_dict["run_start"][:19],
        'Model': get_result_dict['model_info']["type"],
        'Result JSON name': get_result_dict['result_json_name'],
        'Duration (s)': "{:.2f}".format(get_result_dict['run_seconds']),
        'mean_relative_error': "{:.3f} ± {:.3f}".format(
            get_result_dict["results"]["test_score"]["mean_relative_error"],
            get_result_dict["results"]["test_score_sem"]["mean_relative_error"]
        ),
        'mean_absolute_error': "{:.3f} ± {:.3f}".format(
            get_result_dict["results"]["test_score"]["mean_absolute_error"],
            get_result_dict["results"]["test_score_sem"]["mean_absolute_error"]
        ),
        'mean_squared_error': "{:.3f} ± {:.3f}".format(
            get_result_dict["results"]["test_score"]["mean_squared_error"],
            get_result_dict["results"]["test_score_sem"]["mean_squared_error"]
        ),
    }
    assert expected == row
