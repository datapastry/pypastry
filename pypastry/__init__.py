import json
import pkgutil
from datetime import datetime
from glob import glob
from importlib import import_module
from os import path

import pandas as pd
from git import Repo
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate, BaseCrossValidator
from tomlkit.toml_document import TOMLDocument

RESULTS_GLOB = "results/*.json"


def run_experiment(dataset: DataFrame, label_column: str, predictor: BaseEstimator,
                   cross_validator: BaseCrossValidator, scorer):
    print(dataset)
    X = dataset.drop(columns=[label_column])
    y = dataset[label_column]
    predictors = [predictor]

    run_infos = evaluate_predictors(X, predictors, y, cross_validator, scorer)

    repo = Repo('.')
    repo.git.add(update=True)
    repo.index.commit('Run evaluation')
    sha = repo.head.commit.hexsha

    for i, run_info in enumerate(run_infos):
        run_id = sha + '.' + str(i)
        output_path = path.join('results', str(run_id)) + '.json'
        with open(output_path, 'w') as output_file:
            json.dump(run_info, output_file, indent=4)

        repo.index.add([output_path])
        print(run_info)
    repo.index.commit('Add results')
    print_results()


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


def get_dataset(config: TOMLDocument) -> pd.DataFrame:
    data_sources = {}
    for data_source in config['data_source']:
        path = data_source['path']
        print("Getting data from", path)
        data = pd.read_csv(path)
        data_sources[data_source['name']] = data
    return data_sources[config['dataset']['data_source']]


def get_predictors():
    predictors = []

    for _, module_name, _ in pkgutil.walk_packages(['.']):
        # print("Found submodule %s (is a package: %s)" % (modname, ispkg))
        module = import_module(module_name)
        try:
            predictor = module.get_predictor()
        except AttributeError:
            continue
        predictors.append(predictor)
    return predictors


def print_results():
    results = []
    for path in glob(RESULTS_GLOB):
        with open(path) as results_file:
            git_hash = path[8:16]

            result_json = json.load(results_file)
            result = {
                'Git hash': git_hash,
                'Run start': result_json['run_start'][:19],
                'Model': result_json['model_info']['type'],
                'Score': "{:.3f} ± {:.3f}".format(result_json['results']['test_score'],
                                            result_json['results']['test_score_sem']),
                'Duration (s)': "{:.2f}".format(result_json['run_seconds']),
            }
            results.append(result)

    results.sort(key=lambda row: row['Run start'])
    results_dataframe = DataFrame(results)
    print(results_dataframe)


def get_model_info(model: BaseEstimator):
    info = model.get_params()
    info['type'] = type(model).__name__
    return info