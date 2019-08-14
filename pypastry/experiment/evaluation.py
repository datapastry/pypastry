from datetime import datetime
from types import ModuleType
from typing import Any

import pandas as pd
from git import Repo

from pypastry.experiment import Experiment
from pypastry.experiment.hasher import get_dataset_hash
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate

from pypastry.experiment.results import ResultsRepo


class ExperimentRunner:
    def __init__(self, git_repo: Repo, results_repo: ResultsRepo, results_display: ModuleType):
        self.git_repo = git_repo
        self.results_repo = results_repo
        self.results_display = results_display

    def run_experiment(self, experiment: Experiment, force: bool, message: str):
        print("Got dataset with {} rows".format(len(experiment.dataset)))
        if force or self.git_repo.is_dirty():
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
        run_infos = _evaluate_predictors(X, predictors, y, experiment.cross_validator, experiment.scorer)
        self.git_repo.git.add(update=True)
        dataset_hash = get_dataset_hash(experiment.dataset)
        dataset_info = {
            'hash': dataset_hash,
            'columns': experiment.dataset.columns.tolist(),
        }
        new_filenames = self.results_repo.save_results(run_infos, dataset_info)
        self.git_repo.index.add(new_filenames)
        self.git_repo.index.commit(message)


def _evaluate_predictors(X, predictors, y, cross_validator, scorer):
    run_infos = []
    for predictor in predictors:
        start = datetime.utcnow()
        scores_dict = cross_validate(predictor, X, y, cv=cross_validator, scoring=scorer)
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

