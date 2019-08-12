import json
from datetime import datetime
from os import mkdir
from tempfile import NamedTemporaryFile

import pandas as pd
from pypastry.core import print_display
from git import Repo
from pypastry import Experiment
from pypastry.experiment.display import cache_display
from pypastry.experiment.hasher import get_dataset_hash
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate


def run_experiment(experiment: Experiment, force: bool, message: str):
    print("Got dataset with {} rows".format(len(experiment.dataset)))
    repo = Repo('.')
    if force or repo.is_dirty():
        _run_evaluation(experiment.cross_validator, experiment.dataset,
                        experiment.label_column, experiment.predictor, repo, experiment.scorer,
                        message)
        cache_display()
    else:
        print("Clean repo, nothing to do")
    print_display()


def _run_evaluation(cross_validator, dataset, label_column, predictor, repo, scorer, message):
    X = dataset.drop(columns=[label_column])
    y = dataset[label_column]
    predictors = [predictor]
    run_infos = evaluate_predictors(X, predictors, y, cross_validator, scorer)
    repo.git.add(update=True)
    dataset_hash = get_dataset_hash(dataset)
    dataset_info = {
        'hash': dataset_hash,
        'columns': dataset.columns.tolist(),
    }
    try:
        mkdir('results')
    except FileExistsError:
        pass
    for i, run_info in enumerate(run_infos):
        run_info['dataset'] = dataset_info
        with NamedTemporaryFile(mode='w', prefix='result-', suffix='.json', dir='results', delete=False) as output_file:
            json.dump(run_info, output_file, indent=4)
            output_file.flush()
            repo.index.add([output_file.name])
    repo.index.commit(message)


def evaluate_predictors(X, predictors, y, cross_validator, scorer):
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