import json
import os
from glob import glob
from os import mkdir
from tempfile import NamedTemporaryFile
from typing import Dict, Any, List, NamedTuple

from git import Repo


Result = NamedTuple('Result', [('data', Dict[str, Any]), ('git_hash', str)])


class ResultsRepo:
    def __init__(self, results_path: str, git_repo: Repo):
        self.git_repo = git_repo
        self.results_path = results_path

    def save_results(self, run_infos: List[Dict[str, Any]], dataset_info: Dict[str, Any], message: str):
        try:
            mkdir(self.results_path)
        except FileExistsError:
            pass
        for i, run_info in enumerate(run_infos):
            run_info['dataset'] = dataset_info
            with NamedTemporaryFile(mode='w', prefix='result-', suffix='.json',
                                    dir=self.results_path, delete=False) as output_file:
                json.dump(run_info, output_file, indent=4)
                output_file.flush()
                self.git_repo.index.add([output_file.name])
        self.git_repo.index.commit(message)

    def get_results(self):
        results_glob = os.path.join(self.results_path, '*')
        for path in glob(results_glob):
            with open(path) as results_file:
                git_hash = next(self.git_repo.iter_commits(paths=path)).hexsha[:8]
                result_json = json.load(results_file)
                yield Result(result_json, git_hash)
