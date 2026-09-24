"""Reusable football analytics components."""


def __getattr__(name):
    if name == "FootballExperiment":
        from .experiments import FootballExperiment
        return FootballExperiment
    raise AttributeError(name)
