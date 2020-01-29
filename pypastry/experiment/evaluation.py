from datetime import datetime
from types import ModuleType
from typing import Any, Dict

import numpy as np
import pandas as pd
from git import Repo
from joblib import Parallel, delayed
from pandas import Series
from pypastry.experiment import Experiment
from pypastry.experiment.hasher import get_dataset_hash
from pypastry.experiment.results import ResultsRepo
from scipy.stats import sem
from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.model_selection import check_cv, PredefinedSplit


class ExperimentRunner:
    def __init__(self, git_repo: Repo, results_repo: ResultsRepo, results_display: ModuleType):
        self.git_repo = git_repo
        self.results_repo = results_repo
        self.results_display = results_display

    def run_experiment(
        self,
        experiment: Experiment,
        force: bool,
        message: str,
        limit: int
    ):
        print("Got dataset with {} rows".format(len(experiment.dataset)))
        if force or self.git_repo.is_dirty():
            print("Running evaluation")
            self._run_evaluation(experiment, message)
            results = self.results_repo.get_results(self.git_repo)
            self.results_display.cache_display(results)
        else:
            print("Clean repo, nothing to do")
        self.results_display.print_cache_file(limit)

    def _run_evaluation(self, experiment: Experiment, message: str):
        run_info = evaluate_predictor(experiment)
        self.git_repo.git.add(update=True)
        dataset_hash = get_dataset_hash(experiment.dataset, experiment.test_set)
        dataset_info = {
            'hash': dataset_hash,
            'columns': experiment.dataset.columns.tolist(),
        }
        new_filenames = self.results_repo.save_results(run_info, dataset_info)
        self.git_repo.index.add(new_filenames)
        self.git_repo.index.commit(message)


def evaluate_predictor(experiment: Experiment) -> Dict[str, Any]:
    start = datetime.utcnow()

    scores = _get_scores(experiment)

    end = datetime.utcnow()

    scores_array = np.hstack(scores)
    mean_score = scores_array.mean()
    sem_score = sem(scores_array)
    results = {'test_score': mean_score, 'test_score_sem': sem_score}

    model_info = get_model_info(experiment.predictor)

    run_info = {
        'run_start': str(start),
        'run_end': str(end),
        'run_seconds': (end - start).total_seconds(),
        'results': results,
        'model_info': model_info,
    }
    return run_info


def get_model_info(model: BaseEstimator):
    info = model.get_params()
    info['type'] = type(model).__name__
    return info


def _get_scores(experiment: Experiment):
    if experiment.test_set is not None:
        assert experiment.cross_validator is None, "Cannot use a cross validator with train test split"
        dataset = pd.concat([experiment.dataset, experiment.test_set])
        split = np.array([-1] * len(experiment.dataset) + [1] * len(experiment.test_set))
        cross_validator = PredefinedSplit(split)
    else:
        dataset = experiment.dataset
        cross_validator = experiment.cross_validator

    X = dataset.drop(columns=[experiment.label_column])
    y = dataset[experiment.label_column]
    if experiment.group_column is None:
        if experiment.average_scores_on_instances:
            groups = Series(range(len(X)), index=X.index)
        else:
            groups = None
    else:
        groups = X[experiment.group_column]
        X = X.drop(columns=[experiment.group_column])

    cv = check_cv(cross_validator, y, classifier=is_classifier(experiment.predictor))
    train_test = cv.split(X, y, groups)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=None, verbose=False,
                        pre_dispatch='2*n_jobs')
    scores = parallel(
        delayed(_fit_and_predict)(
            clone(experiment.predictor), X, y, train, test, groups, experiment.scorer)
        for train, test in train_test)
    return scores


def _fit_and_predict(estimator: BaseEstimator, X, y, train, test, groups, scorer):
    if groups is not None:
        return _fit_and_predict_groups(X, estimator, groups, scorer, test, train, y)
    else:
        return _fit_and_predict_simple(X, estimator, scorer, test, train, y)


def _fit_and_predict_simple(X, estimator, scorer, test, train, y):
    X_train = X.iloc[train]
    y_train = y.iloc[train]
    estimator.fit(X_train, y_train)
    X_test = X.iloc[test]
    y_test = y.iloc[test]
    score = scorer(estimator, X_test, y_test)
    return [score]


def _fit_and_predict_groups(X, estimator, groups, scorer, test, train, y):
    X_train = X.iloc[train]
    y_train = y.iloc[train]
    estimator.fit(X_train, y_train)
    X_test = X.iloc[test]
    y_test = y.iloc[test]
    groups_test = groups.iloc[test]
    test_df = pd.DataFrame(X_test)
    test_df['y'] = y_test
    test_df['groups'] = groups_test
    # test_df = pd.DataFrame({'X': X_test.values, 'y': y_test.values, 'groups': groups_test.values})
    test_groups = test_df.groupby('groups')
    scores = []
    for key, group in test_groups:
        X_group = group[X.columns]
        score = scorer(estimator, X_group, group['y'])
        scores.append(score)
    print("SCORES", scores)
    return scores

