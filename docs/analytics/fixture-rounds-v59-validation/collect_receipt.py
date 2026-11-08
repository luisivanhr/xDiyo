"""Bind the real dataset integration and regression receipt to exact source bytes."""
import json
import hashlib
from pathlib import Path
import platform
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pandas as pd
import pyarrow
import pytest
from xdiyo_analytics.data.fixture_rounds import file_sha256, load_fixture_rounds

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RELEASE = ROOT / "data/fixture_rounds/v59_20261108"
SOURCES = (
    "src/xdiyo_analytics/data/fixture_rounds.py",
    "src/xdiyo_analytics/data/__init__.py",
    "examples/publish_fixture_rounds_v59.py",
    "examples/integrate_fixture_rounds.py",
    "tests/analytics/test_fixture_rounds.py",
)


if __name__ == "__main__":
    mapping = load_fixture_rounds(RELEASE)
    shutil.copyfile(ROOT / "scratch/round-integration-v59/integration.json", HERE / "integration.json")
    test = HERE / "tests-python312-pandas301.xml"
    tree = ET.parse(test)
    for suite in tree.getroot().iter("testsuite"):
        suite.attrib.pop("hostname", None)
    tree.write(test, encoding="utf-8", xml_declaration=True)
    suites = [dict(s.attrib) for s in tree.getroot().iter("testsuite")]
    if any(int(s.get("failures", 0)) or int(s.get("errors", 0)) for s in suites):
        raise ValueError("Regression suite did not pass")
    output = dict(base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        environment=dict(python=platform.python_version(), pandas=pd.__version__, pyarrow=pyarrow.__version__, pytest=pytest.__version__),
        source_sha256={name: hashlib.sha256((ROOT / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest() for name in SOURCES},
        source_hash_normalization="UTF-8 with LF newlines; portable across Git checkout newline settings",
        receipt_redaction="Hostname removed; test results and timings unchanged",
        release_manifest_sha256=file_sha256(RELEASE / "manifest.json"),
        integration_receipt_sha256=file_sha256(HERE / "integration.json"),
        tests=suites, test_receipt_sha256=file_sha256(test), counts=mapping.provenance["counts"],
        experiment_prepared=False, experiment_run=False, forecast_timing_changed=False)
    (HERE / "verification.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
