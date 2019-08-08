import pkgutil
from importlib import import_module

import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from tomlkit.toml_document import TOMLDocument

RESULTS_GLOB = "results/*.json"


class Experiment:
    def __init__(self, dataset: DataFrame, label_column: str, predictor: BaseEstimator,
                 cross_validator: BaseCrossValidator, scorer):
        self.dataset = dataset
        self.label_column = label_column
        self.predictor = predictor
        self.cross_validator = cross_validator
        self.scorer = scorer


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


