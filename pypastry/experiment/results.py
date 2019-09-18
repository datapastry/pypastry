import json
import os
from glob import glob
from os import mkdir
from tempfile import NamedTemporaryFile
from typing import Dict, Any, List, NamedTuple

from git import Repo

Result = NamedTuple('Result', [('data', Dict[str, Any]), ('git_hash', str)])


class ResultsRepo:
    def __init__(self, results_path: str):
        self.results_path = results_path

    def save_results(self, run_infos: List[Dict[str, Any]], dataset_info: Dict[str, Any]) -> List[str]:
        """
        Stores all experiment results into a json file
        Args:
            run_infos (dict): Dictionary containing results information
            daraset_info (dict): Dictionary contatining dataset information

        Returns:
            List of result files names
        """
        try:
            mkdir(self.results_path)
        except FileExistsError:
            pass
        new_filenames = []
        for i, run_info in enumerate(run_infos):
            run_info['dataset'] = dataset_info
            # TODO put git commit hash or combination of username and timestamp in the prefix to avoid merge conflicts
            with NamedTemporaryFile(mode='w', prefix='result-', suffix='.json',
                                    dir=self.results_path, delete=False) as output_file:
                json.dump(run_info, output_file, indent=4)
                output_file.flush()
            new_filenames.append(output_file.name)
        return new_filenames

    def get_results(self, git_repo):
        results_glob = os.path.join(self.results_path, '*')
        for path in glob(results_glob):
            with open(path) as results_file:
                git_hash = next(git_repo.iter_commits(paths=path)).hexsha[:8]
                result_json = json.load(results_file)
                yield Result(result_json, git_hash)
