import json
from os import mkdir
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Any, List, NamedTuple

Result = NamedTuple('Result', [('data', Dict[str, Any]), ('git_hash', str), ('summary', str)])


class ResultsRepo:
    def __init__(self, results_path: str):
        self.results_path = results_path

    def save_results(self, run_info: Dict[str, Any], dataset_info: Dict[str, Any]) -> List[str]:
        try:
            mkdir(self.results_path)
        except FileExistsError:
            pass
        new_filenames = []
        run_info['dataset'] = dataset_info
        with NamedTemporaryFile(mode='w', prefix='result-', suffix='.json',
                                dir=self.results_path, delete=False) as output_file:
            json.dump(run_info, output_file, indent=4, default=str)
            output_file.flush()
        new_filenames.append(output_file.name)
        return new_filenames

    def get_results(self, git_repo):
        for path in Path(self.results_path).glob('*'):
            with open(path) as results_file:
                git_commit = next(git_repo.iter_commits(paths=path.absolute()))
                summary = git_commit.summary
                git_hash = git_commit.hexsha[:8]
                result_json = json.load(results_file)
                yield Result(result_json, git_hash, summary)
