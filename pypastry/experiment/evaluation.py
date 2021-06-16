import json
from datetime import datetime
from types import ModuleType
from typing import Any, Dict, Tuple, List
from pathlib import Path

import numpy as np
import pandas as pd
from git import Repo
from joblib import Parallel, delayed
from pandas import Series
from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.metrics._scorer import _BaseScorer
from sklearn.model_selection import check_cv, PredefinedSplit

from pypastry import display
from pypastry.experiment import Experiment
from pypastry.experiment.hasher import get_dataset_hash
from pypastry.experiment.results import ResultsRepo
from pypastry.paths import REPO_PATH, RESULTS_PATH

MAX_PARAMETER_VALUE_LENGTH = 500


class DirtyRepoError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ExperimentRunner:
    def __init__(self, git_repo: Repo, results_repo: ResultsRepo, results_display: ModuleType):
        self.git_repo = git_repo
        self.results_repo = results_repo
        self.results_display = results_display

    def run_experiment(
        self,
        experiment: Experiment,
        message: str = "",
        force: bool = False,
        limit: int = None,
        show_results: bool = True,
    ) -> Tuple[List[BaseEstimator], Path]:

        print("Got dataset with {} rows".format(len(experiment.dataset)))
        if force or not self.git_repo.is_dirty():
            print("Running evaluation")
            estimators, result_file_path = self._run_evaluation(experiment, message)
            results = self.results_repo.get_results()
            self.results_display.cache_display(results)
        else:
            raise DirtyRepoError("There are untracked/unstaged/staged changes in git repo, force flag was not given. "
                                 "Please commit your changes or provide force flag - note that in this case "
                                 "saved commit hash in your result file will not correspond to the actual code!")
        if show_results:
            self.results_display.print_cache_file(limit)

        return estimators, result_file_path

    def _run_evaluation(self, experiment: Experiment, message: str) -> Tuple[List[BaseEstimator], Path]:
        run_info, estimators = evaluate_predictor(experiment)
        dataset_hash = get_dataset_hash(experiment.dataset, experiment.test_set)
        dataset_info = {
            'hash': dataset_hash,
            'columns': experiment.dataset.columns.tolist(),
            'size': len(experiment.dataset),
        }
        git_info = {
            "git_hash_msg": ("dirty_" if self.git_repo.is_dirty() else "") + self.git_repo.head.object.hexsha[:8],
            "git_summary_msg": message,
        }
        result_file_path = self.results_repo.save_results(run_info, dataset_info, git_info=git_info)

        return estimators, result_file_path


def evaluate_predictor(experiment: Experiment) -> Dict[str, Tuple[Any, List[BaseEstimator]]]:
    start = datetime.utcnow()
    scores, estimators = _get_scores_and_estimators(experiment)
    end = datetime.utcnow()

    additional = experiment.additional_info
    additional_info = [additional(estimator) if additional is not None else None
                       for estimator in estimators]

    if experiment.group_column is not None:
        for group, score_values in scores:
            score_values[experiment.group_column] = group

    values = [x[1] for x in scores]

    scores_array = pd.DataFrame(values)

    mean_score = scores_array.mean().to_dict()
    sem_score = scores_array.sem().to_dict()
    results = {'test_score': mean_score, 'test_score_sem': sem_score}

    model_info = get_model_info(experiment.predictor)

    run_info = {
        'run_start': str(start),
        'run_end': str(end),
        'run_seconds': (end - start).total_seconds(),
        'results': results,
        'results_detail': scores_array.to_dict('list'),
        'model_info': model_info,
        'additional_info': additional_info,
    }

    return run_info, estimators


def get_model_info(model: BaseEstimator):
    all_info = model.get_params()
    info = {key: value for key, value in all_info.items()
            if len(json.dumps(value, default=str)) < MAX_PARAMETER_VALUE_LENGTH}
    info['type'] = type(model).__name__
    return info


def _get_scores_and_estimators(experiment: Experiment) -> Tuple[List[float], List[Any]]:
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
    scores_and_estimators = parallel(
        delayed(_fit_and_predict)(
            clone(experiment.predictor), X, y, train, test, groups, experiment.scorer)
        for train, test in train_test)
    scores_lists, estimators = zip(*scores_and_estimators)
    scores = [score for score_list in scores_lists for score in score_list]
    return scores, estimators


def _fit_and_predict(estimator: BaseEstimator, X, y, train, test, groups, scorer):
    if groups is not None:
        scores = _fit_and_predict_groups(X, estimator, groups, scorer, test, train, y)
    else:
        scores = _fit_and_predict_simple(X, estimator, scorer, test, train, y)
    return scores, estimator


def _fit_and_predict_simple(X, estimator, scorers, test, train, y):
    X_train = X.iloc[train]
    y_train = y.iloc[train]
    estimator.fit(X_train, y_train)
    X_test = X.iloc[test]
    y_test = y.iloc[test]
    score = _score(scorers, estimator, X_test, y_test)
    return [(None, score)]


def _fit_and_predict_groups(X, estimator, groups, scorers, test, train, y):
    X_train = X.iloc[train]
    y_train = y.iloc[train]
    estimator.fit(X_train, y_train)
    X_test = X.iloc[test]
    y_test = y.iloc[test]
    groups_test = groups.iloc[test]
    test_df = pd.DataFrame(X_test)
    test_df['y'] = y_test
    test_df['groups'] = groups_test
    test_groups = test_df.groupby('groups')
    scores = []
    for key, group in test_groups:
        X_group = group[X.columns]
        score = _score(scorers, estimator, X_group, group['y'])
        scores.append((key, score))
    return scores


def _score(scorers: List[_BaseScorer], estimator, X_test, y_test):
    scores = {}
    for scorer in scorers:
        score = scorer(estimator, X_test, y_test)
        score_name = scorer._score_func.__name__
        sign = scorer._sign
        score_ignoring_sign = score*sign
        scores[score_name] = score_ignoring_sign
    return scores


def run_experiment(experiment, message="", force=False, show_results=True) -> Tuple[List[BaseEstimator], Path]:
    git_repo = Repo(REPO_PATH, search_parent_directories=True)  # type: pypastry.experiment.Experiment
    results_repo = ResultsRepo(RESULTS_PATH)  # type: pypastry.experiment.results.ResultsRepo
    runner = ExperimentRunner(git_repo, results_repo, display)  # type:
    # pypastry.experiment.evaluation.ExperimentRunner
    return runner.run_experiment(
        experiment=experiment,
        message=message,
        force=force,
        show_results=show_results,
    )
