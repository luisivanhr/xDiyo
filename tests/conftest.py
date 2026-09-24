"""Keep test scratch data inside this workspace, without reusing old paths."""
from pathlib import Path
import uuid


def pytest_configure(config):
    if config.option.basetemp is None:
        parent = Path(__file__).resolve().parents[1] / ".pytest_tmp"
        parent.mkdir(exist_ok=True)
        config.option.basetemp = str(parent / uuid.uuid4().hex)

