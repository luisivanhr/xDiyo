"""Model-independent candidate definitions and deterministic grid generation."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import product

from ..experiments import configuration_hash


@dataclass
class Candidate:
    """One complete fitting configuration, not a fitted model.

    model_factory() constructs a fresh ModelAdapter and preprocessing. config is
    a JSON-compatible description for persistence; callers must describe all
    meaningful model/feature/seed choices, including custom callable behavior.
    pre_analysis optionally contains train-scoped PreTrainingAnalysis studies;
    features_from names its consumed selector. They are rerun in each new fitting
    population, including the selected candidate's outer evaluation.

    complexity(model) optionally returns named nonnegative complexity measures.
    fit_statistics(model), or the adapter's fit_statistics() method, may supply
    FitStatistics when information criteria are requested. These inspect fitted
    models; they must not consult evaluation outcomes.
    """
    name: str
    model_factory: object
    config: dict = field(default_factory=dict)
    feature_columns: object = None
    target_columns: object = None
    pre_analysis: object = None
    features_from: str | None = None
    control: object = None
    validation: object = None
    observer: object = None
    complexity: object = None
    fit_statistics: object = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or not callable(self.model_factory):
            raise ValueError("Candidates need a name and a fresh-model factory.")
        if not isinstance(self.config, dict):
            raise TypeError("Candidate config must be an explicit description mapping.")
        configuration_hash(self.config)
        if self.features_from is not None and self.pre_analysis is None:
            raise ValueError("features_from requires candidate pre_analysis.")


@dataclass
class GridCandidates:
    """Iterable Cartesian grid; build(parameters) returns a Candidate.

    parameters is a mapping or a list of mappings for conditional grids. Each
    value is a nonempty sequence. An empty mapping yields one configuration.
    Values are copied before calling build; no sklearn dependency is required.
    Grid values are recorded in config.grid_parameters and names gain a stable
    one-based suffix. build owns model construction and parameter application.
    Other finite Candidate iterables can be supplied directly to ModelSelection.
    """
    parameters: object
    build: object

    def __iter__(self):
        grids = [self.parameters] if isinstance(self.parameters, dict) else list(self.parameters)
        number = 0
        for grid in grids:
            if not isinstance(grid, dict) or any(not isinstance(key, str) for key in grid):
                raise TypeError("Each parameter grid must map parameter names to sequences.")
            choices = []
            for values in grid.values():
                if isinstance(values, (str, bytes, dict)):
                    raise TypeError("Grid parameter values must be sequences of choices.")
                values = list(values)
                if not values:
                    raise ValueError("Grid parameter choices cannot be empty.")
                choices.append(values)
            for combination in product(*choices):
                parameters = deepcopy(dict(zip(grid, combination)))
                candidate = self.build(deepcopy(parameters))
                if not isinstance(candidate, Candidate):
                    raise TypeError("Grid build(parameters) must return Candidate.")
                number += 1
                yield replace(candidate, name=f"{candidate.name} [{number}]",
                              config={**deepcopy(candidate.config), "grid_parameters": parameters})
