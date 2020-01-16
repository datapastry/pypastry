from typing import Any

from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold


class Experiment:
    def __init__(self, dataset: DataFrame, label_column: str, predictor: BaseEstimator,
                 cross_validator: Any = None, scorer: Any = None, group_column: str=None,
                 test_set: DataFrame = None):
        if (test_set is not None) == (cross_validator is not None):
            raise ValueError("You must specify either a cross validator or a test set (and not both)")

        self.dataset = dataset
        self.label_column = label_column
        self.predictor = predictor
        self.cross_validator = cross_validator
        self.scorer = scorer if scorer is not None else make_scorer(accuracy_score)
        self.group_column = group_column
        self.test_set = test_set
