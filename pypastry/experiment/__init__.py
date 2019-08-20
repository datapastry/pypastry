from typing import Any

from pandas import DataFrame
from sklearn.base import BaseEstimator


class Experiment:
    def __init__(self, dataset: DataFrame, label_column: str, predictor: BaseEstimator,
                 cross_validator: Any, scorer, group_column=None):
        self.dataset = dataset
        self.label_column = label_column
        self.predictor = predictor
        self.cross_validator = cross_validator
        self.scorer = scorer
        self.group_column = group_column
