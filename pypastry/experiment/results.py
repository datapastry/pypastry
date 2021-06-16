import json
import os
import glob
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Any, NamedTuple


Result = NamedTuple('Result', [('data', Dict[str, Any])])


class ResultsRepo:
    def __init__(self, results_path: str):
        self.results_path = results_path

    def save_results(self, run_info: Dict[str, Any], dataset_info: Dict[str, Any], git_info: Dict[str, str]) -> Path:
        try:
            os.mkdir(self.results_path)
        except FileExistsError:
            pass
        run_info['dataset'] = dataset_info
        run_info['git_hash'] = git_info["git_hash_msg"]
        run_info['git_summary'] = git_info["git_summary_msg"]
        with NamedTemporaryFile(mode='w', prefix='result-', suffix='.json',
                                dir=self.results_path, delete=False) as output_file:
            result_file_path = Path(output_file.name)
            run_info["result_json_name"] = result_file_path.name
            json.dump(run_info, output_file, indent=4, default=str)
            output_file.flush()

            return result_file_path

    def get_results(self):
        for path in glob.glob(os.path.join(self.results_path, "*.json")):
            with open(str(path), "r") as results_file:
                result_json = json.load(results_file)
            yield Result(result_json)
