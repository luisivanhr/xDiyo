"""Training curves and coefficient inspection without further model fitting."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from .contracts import Artifact, StudyResult
from .studies import _plotting, _style


@dataclass(kw_only=True)
class LearningCurveReporter:
    """Stored training/validation loss curves, separated by fold and attempt.

    model partition inspects fitted artifacts, not test targets. type is explicit
    per_fold or overall; overall displays separate curves, never averages unequal
    training histories. metrics=None displays every stored metric. attempts=None
    displays all attempts; 'selected' restricts to the retained attempt. Native
    histories can be retrospective; absent histories are reported as unavailable.
    """
    type: str
    partition: str = "model"
    metrics: object = None
    attempts: object = None
    supported_types: ClassVar[tuple] = ("per_fold", "overall")

    def run(self, context):
        if context.partition != "model":
            raise ValueError("Learning curves use partition='model'.")
        frames, summaries = [], []
        result = StudyResult("Learning curves")
        for fold_id, fold in context.fold_results.items():
            history = fold.training_history.copy()
            summary = fold.training_summary
            summaries.append(dict(fold_id=fold_id, **{key: value for key, value in summary.items() if key != "attempts"}))
            if history.empty:
                result.notes.append(f"Fold {fold_id}: this adapter exposes no iteration history.")
                continue
            if self.metrics is not None:
                names = [self.metrics] if isinstance(self.metrics, str) else list(self.metrics)
                history = history.loc[history.metric.isin(names)]
            if isinstance(self.attempts, str) and self.attempts == "selected":
                history = history.loc[history.attempt.eq(summary.get("selected_attempt", 0))]
            elif self.attempts is not None:
                if isinstance(self.attempts, str):
                    raise ValueError("attempts must be None, 'selected', or a sequence of attempt IDs.")
                history = history.loc[history.attempt.isin(self.attempts)]
            history["fold_id"] = fold_id
            frames.append(history)
        result.tables["summary"] = pd.DataFrame(summaries)
        # Nested attempt records belong in a separate exportable table.
        result.tables["attempts"] = pd.DataFrame([dict(fold_id=i, **attempt)
                                                 for i, fold in context.fold_results.items()
                                                 for attempt in fold.training_summary.get("attempts", [])])
        table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        result.tables["history"] = table
        if len(table):
            go = _plotting()
            figure = go.Figure()
            for (fold, attempt, metric), values in table.groupby(["fold_id", "attempt", "metric"], sort=False):
                values = values.sort_values("step", kind="stable")
                figure.add_trace(go.Scatter(x=values.step, y=values.value.replace([np.inf, -np.inf], np.nan),
                                            mode="lines+markers", connectgaps=False,
                                            name=f"Fold {fold} / attempt {attempt} / {metric}"))
            result.artifacts.append(Artifact("plotly", _style(figure, "Training histories", "Training step", "Metric value"), "Loss curves"))
        result.artifacts.append(Artifact("table", result.tables["summary"], "Training summary"))
        return result


@dataclass(kw_only=True)
class CoefficientReporter:
    """Signed fitted coefficients for Lasso, Elastic Net and compatible models.

    Adapters expose coefficient_table(): target, label, feature, coefficient,
    term ('feature'/'intercept'). EstimatorAdapter handles coef_ estimators and
    named preprocessing pipelines; custom adapters can supply their own table.
    Nonzero means abs(coefficient)>tolerance. Intercepts are separate from feature
    survival. Tables retain all coefficients even when only surviving bars show.

    Per-fold views use the shared fold selector. Overall views keep individual
    models' coefficients separate and optionally summarize survival frequency.
    The denominator is the number of inspected folds exposing the same output;
    an absent feature contributes no survival. Mean magnitudes are not used to
    compare differently scaled models. Coefficients remain in fitted feature
    space. Adapters declare target units: the target wrapper inverts affine
    target coefficients and explicitly marks nonlinear transformed-target units.
    """
    type: str
    partition: str = "model"
    tolerance: float = 0.0
    include_zeros: bool = False
    survival_frequency: bool = True
    supported_types: ClassVar[tuple] = ("per_fold", "overall")

    def run(self, context):
        if context.partition != "model":
            raise ValueError("Coefficients use partition='model'.")
        if isinstance(self.tolerance, bool) or not np.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative.")
        frames = []
        result = StudyResult("Fitted coefficients", notes=[
            "Coefficient units refer to the features entering the fitted estimator, including preprocessing.",
            f"Survival criterion: absolute coefficient > {self.tolerance:g}. Intercepts are shown separately."])
        for fold_id, model in context.models.items():
            if not callable(getattr(model, "coefficient_table", None)):
                result.notes.append(f"Fold {fold_id}: adapter does not expose coefficient inspection.")
                continue
            try:
                table = model.coefficient_table().copy()
            except (TypeError, AttributeError) as error:
                result.notes.append(f"Fold {fold_id}: coefficient inspection unavailable ({error}).")
                continue
            required = {"target", "label", "feature", "coefficient", "term"}
            if not required <= set(table.columns) or table.duplicated(["target", "label", "feature", "term"]).any():
                raise ValueError("Coefficient tables need distinct target/class/feature terms and required columns.")
            if not table.term.isin(["feature", "intercept"]).all() or not np.isfinite(table.coefficient).all():
                raise ValueError("Coefficient terms must be finite features or intercepts.")
            if 'coefficient_units' in table and table.coefficient_units.eq('transformed_target').any():
                result.notes.append(f"Fold {fold_id}: coefficients are in transformed-target units. "
                                    "The nonlinear inverse has no single original-target slope; predictions still use original units.")
            if 'coefficient_units' in table and table.coefficient_units.eq('log_mean').any():
                result.notes.append(f"Fold {fold_id}: coefficients are in log-mean units. "
                                    "exp(coefficient) is the multiplicative change in expected count for one unit of the fitted feature, holding other inputs fixed. Input scaling still determines that unit.")
            table["fold_id"] = fold_id
            if 'prediction_statistic' in table and table.prediction_statistic.eq('mode').any():
                result.notes.append(f"Fold {fold_id}: predictions use the most probable count (mode); coefficients still describe the underlying log mean, not the integer mode.")
            table["survives"] = table.term.eq("feature") & table.coefficient.abs().gt(self.tolerance)
            frames.append(table)
        columns = ["target", "label", "feature", "coefficient", "term", "fold_id", "survives"]
        all_terms = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        features = all_terms.loc[all_terms.term.eq("feature")].copy()
        intercepts = all_terms.loc[all_terms.term.eq("intercept")].copy()
        result.tables["coefficients"] = all_terms
        result.tables["surviving_coefficients"] = features.loc[features.survives].copy()
        result.tables["intercepts"] = intercepts
        go = _plotting()
        shown = features if self.include_zeros else features.loc[features.survives]
        figure = go.Figure()
        buttons = []
        for (fold_id, target, label), terms in shown.groupby(["fold_id", "target", "label"], dropna=False, sort=False):
            terms = terms.iloc[np.argsort(terms.coefficient.abs().to_numpy(), kind="stable")]
            title = f"Fold {fold_id} · {target}" + (f" · class {label}" if pd.notna(label) else "")
            if 'coefficient_units' in terms and terms.coefficient_units.eq('transformed_target').any():
                title += " · transformed target"
            if 'coefficient_units' in terms and terms.coefficient_units.eq('log_mean').any():
                title += " · log mean"
            figure.add_trace(go.Bar(x=terms.coefficient, y=terms.feature.astype(str), orientation="h",
                                   name=title, visible=len(figure.data)==0,
                                   hovertemplate="%{y}: %{x:.4g}<extra></extra>",
                                   marker_color=np.where(terms.coefficient >= 0, "#58c7b2", "#ed7975")))
            buttons.append(dict(label=title,method='update'))
        if figure.data:
            for i,button in enumerate(buttons):
                button['args']=[{'visible':[j==i for j in range(len(buttons))]}, {'title':button['label']}]
            figure = _style(figure, buttons[0]['label'], "Coefficient (fitted feature units)", "Feature")
            figure.update_layout(height=max(430,min(1200,120+25*max(len(t.y) for t in figure.data))),
                                 margin=dict(l=220,r=35,t=80,b=60),showlegend=False)
            if len(buttons)>1:
                figure.update_layout(updatemenus=[dict(buttons=buttons,direction='down',x=1,y=1.17,xanchor='right')])
            result.artifacts.append(Artifact('plotly',figure,'Surviving feature coefficients'))
        if features.empty or not features.survives.any():
            result.notes.append("No surviving feature coefficients are available in this scope.")
        if self.survival_frequency and len(features):
            rows = []
            for (target, label), output in all_terms.groupby(["target", "label"], dropna=False, sort=False):
                # An intercept-only model still exposes this output and counts
                # as a fold where each absent feature did not survive.
                denominator = output.fold_id.nunique()
                for feature, values in output.loc[output.term.eq("feature")].groupby("feature", sort=False):
                    survived = int(values.survives.sum())
                    rows.append(dict(target=target, label=label, feature=feature, folds_present=values.fold_id.nunique(),
                                     folds_survived=survived, folds_inspected=denominator,
                                     survival_frequency=survived/denominator))
            frequency = pd.DataFrame(rows).sort_values("survival_frequency", ascending=False, kind="stable")
            result.tables["survival_frequency"] = frequency
            result.artifacts.append(Artifact("table", frequency, "Feature survival across inspected folds"))
        result.artifacts.append(Artifact("table", intercepts, "Intercepts"))
        return result
