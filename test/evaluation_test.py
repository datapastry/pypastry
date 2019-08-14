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
    results_display_mock = Mock()
    runner = ExperimentRunner(git_mock, results_repo_mock, results_display_mock)

    runner.run_experiment(experiment, False, "Test commit message")

    # TODO: check the results
    # TODO: check that git commit is called
    # TODO: check that the display is cached
    print(results_repo_mock.save_results.call_args_list)
