import sys

from pypastry.evaluation import run_experiment


def run():
    sys.path.append('.')
    import pie
    experiment = pie.get_experiment()

    run_experiment(experiment)
