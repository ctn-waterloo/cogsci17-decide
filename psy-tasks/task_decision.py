from psyrun import Param

from cogsci17_decide.trial import DecisionTrial

pspace = (
    Param(network=['DriftDiffusion', 'UsherMcClelland']) *
    Param(baseline=[0.1, 0.5, 0.8]) *
    Param(target_sep=[0.05, 0.1, 0.15, 0.2]) *
    Param(noise=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05]) *
    Param(seed=range(10))
)

min_items = 1
max_jobs = None


def execute(**kwargs):
    return DecisionTrial().run(**kwargs)
