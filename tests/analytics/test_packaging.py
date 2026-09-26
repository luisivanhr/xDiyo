"""Exercise the publishable wheel without checkout or editable-install imports."""

import json
from pathlib import Path
import shutil
import site
import subprocess
import sys
import sysconfig
from tempfile import TemporaryDirectory
import textwrap
from zipfile import ZipFile

import pytest


PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT_MODULES = ("__init__", "football", "legacy", "ranking", "recovery", "store")
STATIC_ASSETS = ("index.html", "app.js", "forms.js", "feature-bundles.js", "style.css")


def _run(command, *, cwd, **kwargs):
    result = subprocess.run(command, cwd=cwd, capture_output=True, timeout=120, **kwargs)
    assert result.returncode == 0, (result.stdout, result.stderr)
    return result.stdout


@pytest.fixture(scope="module")
def git_checkout():
    if shutil.which("git") is None:
        pytest.skip("Packaging publication checks require a Git checkout.")
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=PROJECT,
                            capture_output=True, text=True, timeout=15)
    if result.returncode or Path(result.stdout.strip()).resolve() != PROJECT:
        pytest.skip("Packaging publication checks require a Git checkout.")
    return PROJECT


def test_experiment_source_is_not_ignored(git_checkout):
    paths = [f"src/xdiyo_analytics/experiments/{name}.py" for name in EXPERIMENT_MODULES]
    assert all((git_checkout / path).is_file() for path in paths)
    # --no-index also catches a broad ignore rule reintroduced after initial staging.
    result = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"],
                            input="\n".join(paths) + "\n", cwd=git_checkout,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 1, f"Package source must be publishable: {result.stdout}{result.stderr}"


def _copy_publishable_source(project, destination):
    """Include candidate analytics edits, but no unrelated local utility changes."""
    paths = _run(["git", "ls-files", "--cached", "--others", "--exclude-standard",
                  "-z", "--", "src/xdiyo_analytics"], cwd=project).decode().split("\0")
    for relative in {path for path in paths if path} | {"pyproject.toml"}:
        source = project / relative
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    # utils is an explicitly declared legacy package. Its index versions preserve
    # the publication boundary when a local collector refactor is also in progress.
    utilities = _run(["git", "ls-files", "-z", "--", "utils/*.py"], cwd=project)
    for relative in filter(None, utilities.decode().split("\0")):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_run(["git", "show", f":{relative}"], cwd=project))


INSTALLED_SMOKE = textwrap.dedent("""
    import contextlib
    import importlib
    import importlib.metadata
    import io
    import json
    from pathlib import Path
    import sys
    from urllib.request import urlopen

    installation, dependency_paths, workspace = map(json.loads, sys.argv[1:])
    installation = Path(installation).resolve()
    # -I -S omits the working directory, PYTHONPATH and site/.pth processing.
    # Add dependency directories as plain paths, without executing editable hooks.
    sys.path[:0] = [str(installation), *dependency_paths]
    assert not any('editable' in type(finder).__module__ for finder in sys.meta_path)

    import numpy as np
    import pandas as pd
    from xdiyo_analytics import FootballExperiment
    from xdiyo_analytics.datasets import ModelDataset
    from xdiyo_analytics.experiments import PreparedExperiment
    from xdiyo_analytics.selection import Candidate
    from xdiyo_analytics.splits import Fold, SplitPlan

    for name in ('football', 'legacy', 'ranking', 'recovery', 'store'):
        importlib.import_module('xdiyo_analytics.experiments.' + name)

    class MeanModel:
        def fit(self, context):
            self.mean = context.y.mean()

        def predict(self, context):
            return {'predict': pd.DataFrame({name: value for name, value in self.mean.items()},
                                            index=context.X.index)}

    dataset = ModelDataset(pd.DataFrame({'x': [0., 1., 2.]}),
                           pd.DataFrame({'goals': [1., 3., 4.]}),
                           pd.DataFrame({'event_id': [1, 2, 3]}),
                           'match', ('event_id',), ('event_id',), 'home')
    split = SplitPlan([Fold(np.array([0, 1]), np.array([2]), np.array([2]))],
                      3, np.arange(3))
    prepared = PreparedExperiment(dataset, split)
    experiment = FootballExperiment('Installed wheel', output_dir=workspace,
                                    prepare=lambda: prepared)
    assert experiment.prepare() is prepared
    result = experiment.run(model=Candidate('Mean', MeanModel))
    assert result.training.folds[0].predictions['predict'].iloc[0, 0] == 2.
    loaded = experiment.load(result.record['run_id'])
    pd.testing.assert_frame_equal(loaded.dataset.X, dataset.X)
    assert loaded.reused

    distribution = importlib.metadata.distribution('xdiyo-analytics')
    commands = {entry.name: entry for entry in distribution.entry_points
                if entry.group == 'console_scripts'}
    assert set(commands) == {'xdiyo-ui'}
    entry = commands['xdiyo-ui']
    assert entry.value == 'xdiyo_analytics.ui.server:main'
    sys.argv = ['xdiyo-ui', '--help']
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        try:
            entry.load()()
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError('The UI entry point did not handle --help.')
    assert '--no-browser' in output.getvalue()

    from xdiyo_analytics.ui import launch_ui
    from xdiyo_analytics.ui.inventory import inventory
    assert inventory()['components']
    handle = launch_ui(workspace=workspace, open_browser=False)
    try:
        for name in ('index.html', 'app.js', 'forms.js', 'feature-bundles.js',
                     'style.css', 'report.css'):
            with urlopen(handle.url.split('#')[0] + name, timeout=10) as response:
                assert response.status == 200
                assert len(response.read()) > 50
    finally:
        handle.close()
        handle.thread.join(timeout=5)
    for name, module in list(sys.modules.items()):
        if name == 'xdiyo_analytics' or name.startswith(('xdiyo_analytics.', 'utils.')):
            assert Path(module.__file__).resolve().is_relative_to(installation), name
    print('Installed experiment run/reload and UI entry point/assets passed.')
""")


def test_publishable_wheel_runs_without_editable_install(git_checkout):
    # Do not put builds, installed copies or generated experiment runs in source.
    with TemporaryDirectory(prefix="xdiyo-packaging-") as temporary:
        scratch = Path(temporary)
        source = scratch / "source"
        source.mkdir()
        _copy_publishable_source(git_checkout, source)
        wheel_dir = scratch / "wheels"
        _run([sys.executable, "-I", "-m", "pip", "wheel", "--no-deps", "--no-index",
              "--no-build-isolation", "--no-cache-dir", "--disable-pip-version-check", "--wheel-dir",
              str(wheel_dir), str(source)], cwd=scratch)
        wheels = list(wheel_dir.glob("*.whl"))
        assert len(wheels) == 1
        with ZipFile(wheels[0]) as archive:
            names = set(archive.namelist())
        assert {f"xdiyo_analytics/experiments/{name}.py" for name in EXPERIMENT_MODULES} <= names
        assert {f"xdiyo_analytics/ui/static/{name}" for name in STATIC_ASSETS} <= names
        assert "xdiyo_analytics/ui/inventory.json" in names
        assert not any(name.startswith(("scraping/", "data/", "experiments/", "experiment/"))
                       or "_collection" in Path(name).parts or name.endswith(".pyc") for name in names)
        installation = scratch / "installed"
        _run([sys.executable, "-I", "-m", "pip", "install", "--no-deps", "--no-index",
              "--no-compile", "--no-cache-dir", "--disable-pip-version-check", "--target", str(installation),
              str(wheels[0])], cwd=scratch)
        dependencies = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib"),
                               *(path for path in site.getsitepackages()
                                 if Path(path).name in {"site-packages", "dist-packages"})})
        result = _run([sys.executable, "-I", "-S", "-c", INSTALLED_SMOKE,
                       json.dumps(str(installation)), json.dumps(dependencies),
                       json.dumps(str(scratch / "example-run"))], cwd=scratch, text=True)
        assert "Installed experiment run/reload and UI entry point/assets passed." in result
        print(f"Verified wheel: {len(names)} members, {wheels[0].stat().st_size} bytes. {result.strip()}")
