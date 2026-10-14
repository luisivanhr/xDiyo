from dataclasses import replace
from pathlib import Path
import json
from xdiyo_analytics.ui.recipe import catalog_for_ui
from xdiyo_analytics.training import TrainingControl

def full_totals_arms(base_config, frozen_bayesian_columns):
    """Construct only; caller supplies its frozen inputs and exact 11 Bayesian columns."""
    columns=tuple(frozen_bayesian_columns)
    if len(columns)!=11 or len(set(columns))!=11:
        raise ValueError('Supply the frozen 11 unique Bayesian fixture columns.')
    manifest=json.loads(Path(__file__).with_name('full_totals_auxiliary.json').read_text())
    auxiliary=catalog_for_ui().build(manifest['forecast'])
    base=replace(base_config, statistics_forecast=auxiliary,
                 boundary_memory=None, heatmaps=None, fixture_features=())
    return {'totals':base, 'totals_bayesian':replace(base, fixture_features=columns)}

def selection_control(**kwargs):
    """CE-only selection; supply separately chosen max_steps/patience/seed."""
    if 'monitor' in kwargs or 'direction' in kwargs:
        raise ValueError('These arms select solely by validation cross entropy.')
    return TrainingControl(monitor='validation_loss', direction='minimize', **kwargs)
