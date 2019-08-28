import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from pypastry.experiment import Experiment


def get_experiment():
    dataset = pd.DataFrame({
        'feature': [1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        'class': [True, False, True, True, False, False, True, True, False, False],
    })
    predictor = DecisionTreeClassifier()
    cross_validator = StratifiedKFold(n_splits=5)
    scorer = make_scorer(f1_score)
    label_column = 'class'
    return Experiment(dataset, label_column, predictor, cross_validator, scorer)
