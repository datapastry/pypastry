import json

from pypastry.display import _get_results_dataframe
from pypastry.experiment.results import Result


RESULTS = {
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
        ]
    },
}


def test_get_display():
    result = Result(RESULTS, 'abc123')
    results_dataframe = _get_results_dataframe([result])
    row = results_dataframe.iloc[0].to_dict()

    expected = {
        'Git hash': 'abc123',
        'Dataset hash': '998c9dea',
        'Run start': '2020-01-31 09:19:44',
        'Model': 'KNearestNeighbor',
        'Duration (s)': '4.48',
        'mean_relative_error': '0.500 ± 0.010',
        'mean_absolute_error': '100.000 ± 1.000',
        'mean_squared_error': '1000.000 ± 10.000',
    }
    assert expected == row
