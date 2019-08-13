from pandas import DataFrame
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from pypastry.experiment import Experiment


def test_simple_evaluation():
    dataset = DataFrame({
        'a': [1, 1, 0, 0],
        'b': [1, 1, 0, 0],
    })

    cross_validation = StratifiedShuffleSplit(n_splits=1, test_size=0.5)

    predictor = DecisionTreeClassifier()

    scorer = make_scorer(accuracy_score)

    experiment = Experiment(dataset, 'b', predictor, cross_validation, scorer)

    # TODO: allow passing in a git repo here so that we can easily mock it
    # TODO: also pass in a results creator so that we don't actually create results in tests
    # TODO: pass in a display class that allows no caching
