"""Local, notebook-launchable experiment builder and portable recipes."""

from .catalog import Catalog
from .recipe import (catalog_for_ui, default_recipe, export_notebook, export_python,
                     predict_recipe, prepare_recipe, read_recipe, run_recipe)


def launch_ui(**kwargs):
    """Start the local browser builder; return a handle with url/show/close."""
    from .server import launch_ui as launch
    return launch(**kwargs)


__all__ = ['Catalog', 'catalog_for_ui', 'default_recipe', 'export_notebook', 'export_python',
           'predict_recipe', 'prepare_recipe', 'read_recipe', 'run_recipe', 'launch_ui']
