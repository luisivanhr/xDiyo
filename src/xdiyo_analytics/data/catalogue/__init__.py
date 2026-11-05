"""Explicit dataset selection; legacy loading remains unchanged."""
from .registry import Registry
from .storage import list_datasets, inspect_coverage, publish_dataset, load_snapshot
from .selection import select_data, resolve_selection, save_selection, selected_history, requirement_closure
from .importers import read_historical_results, read_results_workbook, combine_imports

__all__=['Registry','list_datasets','inspect_coverage','publish_dataset','load_snapshot','select_data','resolve_selection',
         'save_selection','selected_history','requirement_closure','read_historical_results','read_results_workbook','combine_imports']

from .migration import plan_recipe_dataset_migration, apply_recipe_dataset_migration
__all__ += ["plan_recipe_dataset_migration", "apply_recipe_dataset_migration"]

from .movements import enrich_movements
__all__ += ['enrich_movements']
