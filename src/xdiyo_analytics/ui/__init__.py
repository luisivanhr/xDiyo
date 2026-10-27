"""Local, notebook-launchable experiment builder and portable recipes."""

from .countdown_migration import migrate_season_countdown_recipe
from .catalog import Catalog
from .refresh import refresh_recipe_data
from .recipe import (catalog_for_ui, default_recipe, export_notebook, export_python,
                     predict_recipe, prepare_recipe, read_recipe, run_recipe)


def launch_ui(**kwargs):
    """Start the local browser builder; return a handle with url/show/close."""
    from .server import launch_ui as launch
    return launch(**kwargs)


__all__ = ['migrate_season_countdown_recipe', 'Catalog', 'catalog_for_ui', 'default_recipe', 'export_notebook', 'export_python',
           'predict_recipe', 'prepare_recipe', 'read_recipe', 'run_recipe', 'launch_ui', 'refresh_recipe_data']
