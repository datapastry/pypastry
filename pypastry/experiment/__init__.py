from typing import Any, Callable, Union, Iterable

from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics._scorer import _BaseScorer as BaseScorer


class Experiment:
    def __init__(self, dataset: DataFrame, label_column: str, predictor: BaseEstimator,
                 cross_validator: Any = None, scorer: Union[BaseScorer, Iterable[BaseScorer]] = None,
                 group_column: str=None, test_set: DataFrame = None, average_scores_on_instances: bool = False,
                 additional_info: Callable[[BaseEstimator], Any] = None):
        if (test_set is not None) == (cross_validator is not None):
            raise ValueError("You must specify either a cross validator or a test set (and not both)")

        if average_scores_on_instances and group_column is not None:
            raise ValueError("You can only average on instances when not grouping instances")

        if scorer is None:
            scorer = [make_scorer(accuracy_score)]

        if not isinstance(scorer, Iterable):
            if not isinstance(scorer, BaseScorer):
                raise ValueError("Scorer must be created using make_scorer()")

            scorer = [scorer]

        self.dataset = dataset
        self.label_column = label_column
        self.predictor = predictor
        self.cross_validator = cross_validator
        self.scorer = scorer
        self.group_column = group_column
        self.test_set = test_set
        self.average_scores_on_instances = average_scores_on_instances
        self.additional_info = additional_info
