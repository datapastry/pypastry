from unittest.mock import Mock, MagicMock

import pytest
from pandas import DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, make_scorer, precision_score
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from pypastry.experiment import Experiment
from pypastry.experiment.evaluation import ExperimentRunner, evaluate_predictor, DirtyRepoError


@pytest.fixture
def simple_dataset():
    return DataFrame({
        'a': [1, 1, 0, 0],
        'b': [1, 1, 0, 0],
    })


@pytest.fixture
def grouped_dataset():
    label = [i % 2 for i in range(100)]
    return DataFrame({
        'a': label,
        'b': label,
        'g': [i // 2 for i in range(100)]
    })


@pytest.fixture
def get_predictor():
    return DecisionTreeClassifier()


@pytest.fixture
def get_cross_validator():
    return StratifiedShuffleSplit(n_splits=1, test_size=0.5)


@pytest.fixture
def get_scorer():
    return make_scorer(accuracy_score)


@pytest.mark.parametrize("dirty, force", [(False, False), (False, True), (True, False), (True, True)])
def test_simple_evaluation(dirty, force, simple_dataset):

    cross_validation = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    predictor = DecisionTreeClassifier()
    scorer = make_scorer(accuracy_score)

    experiment = Experiment(simple_dataset, 'b', predictor, cross_validation, scorer)

    git_mock = Mock()
    git_mock.is_dirty.return_value = dirty
    git_mock.head.object.hexsha = MagicMock()
    results_repo_mock = Mock()
    results_display_mock = Mock()
    runner = ExperimentRunner(git_mock, results_repo_mock, results_display_mock)

    try:
        runner.run_experiment(experiment, "msg", force)
    except DirtyRepoError:
        if dirty is True and force is False:
            # Expected behaviour.
            return
        else:
            raise

    call_args_list = results_repo_mock.save_results.call_args_list
    assert 1 == len(call_args_list)
    run_info, dataset_info = call_args_list[0][0]
    print("Run info", run_info)

    results = run_info['results']
    assert {'accuracy_score': 1.0} == results['test_score']
    assert ['a', 'b'] == dataset_info['columns']

    # TODO: check the hash. Need to find a way to make this consistent between python versions etc.
    # assert '28ea628a50a47c726a9b0ec437c88fc4742d81fd' == dataset_info['hash']

    assert 1 == len(results_display_mock.cache_display.call_args_list)
    print(results_display_mock.cache_display.call_args[0])
    assert len(results_display_mock.cache_display.call_args[0]) > 0
    assert 1 == len(results_display_mock.print_cache_file.call_args_list)


@pytest.mark.parametrize("dirty, force", [(False, False), (False, True), (True, False), (True, True)])
def test_grouped_evaluation(dirty, force, grouped_dataset):

    cross_validation = GroupShuffleSplit(n_splits=1, test_size=0.5)
    predictor = DummyClassifier(strategy='constant', constant=1)
    scorer = make_scorer(accuracy_score)

    experiment = Experiment(grouped_dataset, 'b', predictor, cross_validation, scorer, group_column='g')

    git_mock = Mock()
    git_mock.is_dirty.return_value = dirty
    git_mock.head.object.hexsha = MagicMock()
    results_repo_mock = Mock()
    results_display_mock = Mock()
    runner = ExperimentRunner(git_mock, results_repo_mock, results_display_mock)

    try:
        runner.run_experiment(experiment, "msg", force)
    except DirtyRepoError:
        if dirty is True and force is False:
            # Expected behaviour.
            return
        else:
            raise

    assert 1 == len(results_repo_mock.save_results.call_args_list)
    run_info, dataset_info = results_repo_mock.save_results.call_args[0]

    print("Run infos", run_info)

    expected_results = {'test_score': {'accuracy_score': 0.5}, 'test_score_sem': {'accuracy_score': 0.0}}
    pruned_results = {
        'test_score': {
            'accuracy_score': run_info['results']["test_score"]["accuracy_score"]
        },
        'test_score_sem': {
            'accuracy_score': run_info['results']["test_score_sem"]["accuracy_score"]
        }
    }
    assert expected_results == pruned_results


def test_multiple_scorers(simple_dataset):

    cross_validation = StratifiedShuffleSplit(n_splits=2, test_size=0.5)
    predictor = DummyClassifier(strategy='constant', constant=1)
    scorer = [make_scorer(accuracy_score), make_scorer(precision_score)]

    experiment = Experiment(simple_dataset, 'b', predictor, cross_validation, scorer)

    run_info, _ = evaluate_predictor(experiment)
    results = run_info['results']
    print("Results", results)

    expected_results = {
        'test_score': {'accuracy_score': 0.5, 'precision_score': 0.5},
        'test_score_sem': {'accuracy_score': 0.0, 'precision_score': 0.0},
    }

    assert expected_results == results
