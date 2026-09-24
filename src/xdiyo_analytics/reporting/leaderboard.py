"""Across-run leaderboard using only persisted numerical experiment records."""

from dataclasses import dataclass, field
from typing import ClassVar
from html import escape
import json

from ..experiments import rank_runs
from .contracts import Artifact, StudyResult


@dataclass(kw_only=True)
class ExperimentLeaderboardReporter:
    """Compare final run results, optionally exposing internal search trials.

    Intrinsically overall/experiment scope. Numerical ranking is separately
    available through rank_runs for future model selection. weights may contain
    any number of metrics; use selectors to distinguish targets/configurations.
    Raw, utility and weighted contribution columns are all retained/exported.
    include_trials=True enables a detailed view; run_group can restrict it to one
    experiment run. Legacy records without an explicit role count as final.
    """
    weights: dict
    selectors: dict = field(default_factory=dict)
    scaling: str = "percentile"
    reference_scales: dict = field(default_factory=dict)
    directions: dict = field(default_factory=dict)
    include_trials: bool = False
    run_group: str | None = None
    type: str = "overall"
    partition: str = "experiment"
    supported_types: ClassVar[tuple] = ("overall",)

    def run(self, context):
        if self.partition != "experiment" or context.experiment is None:
            raise ValueError("The experiment leaderboard requires an ExperimentStore.")
        records = [record for record in context.experiment.read_runs()
                   if (self.include_trials or record.get("role", "final") == "final") and
                   (self.run_group is None or record.get("run_group", record["run_id"]) == self.run_group)]
        table = rank_runs(records, self.weights, selectors=self.selectors,
                          scaling=self.scaling, reference_scales=self.reference_scales, directions=self.directions)
        identities = {record["run_id"]: record for record in records}
        table["role"] = [identities[key].get("role", "final") for key in table.run_id]
        table["run_group"] = [identities[key].get("run_group", key) for key in table.run_id]
        table["selected_trial_id"] = [identities[key].get("selected_trial_id") for key in table.run_id]
        bars = [name for name in table if name.startswith("contribution::")] + ["score"]
        details = "".join(
            f'<details><summary>{escape(record["name"])}</summary><pre>'
            f'{escape(json.dumps(record.get("config", {}), ensure_ascii=False, indent=2))}</pre></details>'
            for record in records)
        return StudyResult("Experiment leaderboard", [Artifact("leaderboard", table, "Run comparison", {
            "coefficient_columns": bars,
            "legend": "Higher score is better within each comparison group. Bars show contributions on a 0–1 scale."}),
            Artifact("html", details, "Run configurations")],
            {"leaderboard": table}, [
                "Search trials included." if self.include_trials else "Final results only; internal search trials are hidden.",
                f"Scaling: {self.scaling}. Weights are normalized over positive entries.",
                "Each comparison group has matching metric definitions and evaluated observations; compare ranks within groups.",
                "Missing/undefined required metrics leave runs unranked; weights are never renormalized per run.",
                "Percentile scores depend on the current set of comparable runs; fixed reference scales remain fixed."])
