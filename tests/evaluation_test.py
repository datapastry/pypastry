from unittest.mock import Mock

from pandas import DataFrame
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from pypastry.experiment import Experiment
from pypastry.experiment.evaluation import ExperimentRunner


def test_simple_evaluation():
    dataset = DataFrame({
        'a': [1, 1, 0, 0],
        'b': [1, 1, 0, 0],
    })

    cross_validation = StratifiedShuffleSplit(n_splits=1, test_size=0.5)

    predictor = DecisionTreeClassifier()

    scorer = make_scorer(accuracy_score)

    experiment = Experiment(dataset, 'b', predictor, cross_validation, scorer)

    git_mock = Mock()
    results_repo_mock = Mock()
    new_results_files = ['results/abc.json']
    results_repo_mock.save_results.return_value = new_results_files
    results_display_mock = Mock()
    runner = ExperimentRunner(git_mock, results_repo_mock, results_display_mock)

    commit_message = "Test commit message"
    runner.run_experiment(experiment, False, commit_message)

    # TODO: check the results
    # TODO: check that git commit is called
    # TODO: check that the display is cached

    results_repo_mock.save_results.assert_called_once()
    call_args = results_repo_mock.save_results.call_args[0]
    run_infos, dataset_info = call_args

    assert len(run_infos) == 1

    results = run_infos[0]['results']
    assert 1.0 == results['test_score']
    assert ['a', 'b'] == dataset_info['columns']
    assert '28ea628a50a47c726a9b0ec437c88fc4742d81fd' == dataset_info['hash']

    git_mock.git.add.assert_called_once_with(update=True)
    git_mock.index.add.assert_called_once_with(new_results_files)
    git_mock.index.commit.assert_called_once_with(commit_message)

    results_display_mock.cache_display.assert_called_once()
    print(results_display_mock.cache_display.call_args[0])
    assert len(results_display_mock.cache_display.call_args[0]) > 0
    results_display_mock.print_cache_file.assert_called_once()
