import json
import os
import glob
from tempfile import NamedTemporaryFile
from typing import Dict, Any, List, NamedTuple


Result = NamedTuple('Result', [('data', Dict[str, Any]), ("dirty", bool)])


class ResultsRepo:
    def __init__(self, results_path: str):
        self.results_path = results_path

    def save_results(self, run_info: Dict[str, Any], dataset_info: Dict[str, Any], git_info: Dict[str, str]) -> List[str]:
        try:
            os.mkdir(self.results_path)
        except FileExistsError:
            pass
        new_filenames = []
        run_info['dataset'] = dataset_info
        run_info['git_hash'] = git_info["git_hash_msg"]
        run_info['git_summary'] = git_info["git_summary_msg"]
        with NamedTemporaryFile(mode='w', prefix='result-', suffix='.json',
                                dir=self.results_path, delete=False) as output_file:
            json.dump(run_info, output_file, indent=4, default=str)
            output_file.flush()
        new_filenames.append(output_file.name)
        return new_filenames

    def get_results(self, git_repo):
        for path in glob.glob(os.path.join(self.results_path, "*.json")):
            with open(str(path), "r") as results_file:
                result_json = json.load(results_file)
            yield Result(result_json, git_repo.is_dirty())
