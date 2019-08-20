from datetime import datetime
from types import ModuleType
from typing import Any

import pandas as pd
from git import Repo
from joblib import Parallel, delayed
from sklearn.metrics.scorer import _check_multimetric_scoring

from pypastry.experiment import Experiment
from pypastry.experiment.hasher import get_dataset_hash
from sklearn.base import BaseEstimator, is_classifier, clone
from sklearn.model_selection import cross_validate, cross_val_predict, check_cv

from pypastry.experiment.results import ResultsRepo


class ExperimentRunner:
    def __init__(self, git_repo: Repo, results_repo: ResultsRepo, results_display: ModuleType):
        self.git_repo = git_repo
        self.results_repo = results_repo
        self.results_display = results_display

    def run_experiment(self, experiment: Experiment, force: bool, message: str):
        print("Got dataset with {} rows".format(len(experiment.dataset)))
        if force or self.git_repo.is_dirty():
            print("Running evaluation")
            self._run_evaluation(experiment, message)
            results = self.results_repo.get_results(self.git_repo)
            self.results_display.cache_display(results)
        else:
            print("Clean repo, nothing to do")
        self.results_display.print_cache_file()

    def _run_evaluation(self, experiment: Experiment, message: str):
        X = experiment.dataset.drop(columns=[experiment.label_column])
        y = experiment.dataset[experiment.label_column]
        predictors = [experiment.predictor]
        if experiment.group_column is None:
            groups = None
        else:
            groups = X[experiment.group_column]
            X = X.drop(columns=[experiment.group_column])
        run_infos = _evaluate_predictors(X, predictors, y, groups, experiment.cross_validator, experiment.scorer)
        self.git_repo.git.add(update=True)
        dataset_hash = get_dataset_hash(experiment.dataset)
        dataset_info = {
            'hash': dataset_hash,
            'columns': experiment.dataset.columns.tolist(),
        }
        new_filenames = self.results_repo.save_results(run_infos, dataset_info)
        self.git_repo.index.add(new_filenames)
        self.git_repo.index.commit(message)


def _evaluate_predictors(X, predictors, y, groups, cross_validator, scorer):
    run_infos = []
    for predictor in predictors:
        start = datetime.utcnow()

        predictions = _get_predictions(X, y, groups, cross_validator, predictor)

        # scores_dict = cross_validate(predictor, X, y, groups, cv=cross_validator, scoring=scorer)
        end = datetime.utcnow()

        scores = pd.DataFrame(scores_dict)
        mean_scores = scores.mean()
        sem_scores = scores.sem()
        results = dict(mean_scores.items())
        results.update({k + '_sem': v for k, v in sem_scores.items()})

        model_info = get_model_info(predictor)

        run_info = {
            'run_start': str(start),
            'run_end': str(end),
            'run_seconds': (end - start).total_seconds(),
            'results': results,
            'model_info': model_info,
        }
        run_infos.append(run_info)
    return run_infos


def get_model_info(model: BaseEstimator):
    info = model.get_params()
    info['type'] = type(model).__name__
    return info


def _get_predictions(X, y, groups, cv, estimator):
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=None, verbose=False,
                        pre_dispatch='2*n_jobs')
    scores = parallel(
        delayed(_fit_and_predict)(
            clone(estimator), X, y, train, test)
        for train, test in cv.split(X, y, groups))
    return scores


def _fit_and_predict(estimator, X, y, train, test):
    pass    # TODO: implement

