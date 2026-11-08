"""Verify the combined round/identity tree without publishing data or running models."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from xdiyo_analytics.data import load_fixture_rounds
from xdiyo_analytics.data.catalogue import load_reviewed_team_directory
from xdiyo_analytics.data.catalogue.registry import name_key
from xdiyo_analytics.reporting import TeamCatalog

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ROUND_PARENT = "f7d94c666418a502ff2d495ab2dd841a0e5c8e49"
TEAM_PARENT = "8a20df9933a34aeb7ca7f4b3c43761fd5bf19fea"
BASE = "ec19875ad13de83187a42762e9094e016c778042"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify():
    unchanged = {
        ROUND_PARENT: ["data/fixture_rounds", "src/xdiyo_analytics/data/fixture_rounds.py",
                       "tests/analytics/test_fixture_rounds.py", "examples/integrate_fixture_rounds.py",
                       "examples/publish_fixture_rounds_v59.py"],
        TEAM_PARENT: ["src/xdiyo_analytics/data/catalogue", "docs/analytics/team_assets",
                      "examples/import_named_datasets.py", "pyproject.toml"],
    }
    for parent, paths in unchanged.items():
        git("diff", parent, "--exit-code", "--", *paths)
    untouched = ["data/xDiyo_data", "src/xdiyo_analytics/graphs",
                 "src/xdiyo_analytics/features", "src/xdiyo_analytics/training",
                 "src/xdiyo_analytics/experiments"]
    git("diff", BASE, "--exit-code", "--", *untouched)
    if git("diff", "--name-only", "--diff-filter=U").strip():
        raise ValueError("Unresolved merge conflicts")
    mapping = load_fixture_rounds(ROOT / "data/fixture_rounds/v59_20261108")
    directory = load_reviewed_team_directory()
    aliases = defaultdict(list)
    for team in directory["teams"]:
        for alias in team["names"]:
            aliases[(team["country"], name_key(alias["name"]))].append(
                (alias["start"], alias["end"], team["team_id"]))
    sides = Counter()
    canonical = set()
    fixture_keys = set()
    for fixture in mapping.selections.itertuples():
        pair = []
        for side in ("home", "away"):
            identities = {tid for start, end, tid in aliases[
                (fixture.country, name_key(getattr(fixture, side + "_team")))]
                if start <= fixture.kickoff_date <= end}
            sides[len(identities)] += 1
            canonical.update(identities)
            if len(identities) == 1:
                pair.append(next(iter(identities)))
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(f"Missing/ambiguous/self-match identity: {fixture.match_id}")
        key = (fixture.country, fixture.league, fixture.season, fixture.kickoff_date, *pair)
        if key in fixture_keys:
            raise ValueError(f"Dated canonical fixture collision: {fixture.match_id}")
        fixture_keys.add(key)
    if sides != {1: 122976}:
        raise ValueError(f"Nonunique or missing dated team identities: {sides}")
    assets = ROOT / "docs/analytics/team_assets"
    current = json.loads((assets / "catalog.json").read_text(encoding="utf-8"))
    before = json.loads(git("show", "2a0c0f9cdb1c1fc133f2baf90027692ccfac15d0:docs/analytics/team_assets/catalog.json"))
    for key, entry in before.items():
        if current[key] != entry or sha(assets / entry["badge_path"]) != entry["sha256"]:
            raise ValueError(f"Preexisting catalogue entry or badge changed: {key}")
    catalog = TeamCatalog.builtin()
    formats = Counter()
    for team in directory["teams"]:
        tid = team["team_id"]
        name, image, error = catalog.display(tid, badges=True)
        entry = current[str(tid)]
        if name != team["canonical_name"] or not image or error or sha(assets / entry["badge_path"]) != entry["sha256"]:
            raise ValueError(f"Supplied badge or identity changed: {tid}")
        formats[entry["format"]] += 1
    tests = []
    for filename in ("tests-python312-pandas301.xml", "tests-python314-pandas233.xml"):
        path = HERE / filename
        tree = ET.parse(path)
        for suite in tree.getroot().iter("testsuite"):
            suite.attrib.pop("hostname", None)
            if int(suite.get("errors", 0)) or int(suite.get("failures", 0)):
                raise ValueError(f"Failed test receipt: {filename}")
        tree.write(path, encoding="utf-8", xml_declaration=True)
        tests.append(dict(file=filename, sha256=sha(path), suites=[dict(s.attrib) for s in tree.getroot().iter("testsuite")]))
    result = dict(schema=1, branch="codex/v59-rounds-team-identities",
        merge_parents=[ROUND_PARENT, TEAM_PARENT], shared_base=BASE,
        source_trees_unchanged=unchanged, textual_merge_conflicts=0,
        unrelated_data_and_model_paths_unchanged_from_base=untouched,
        round_manifest_sha256=sha(ROOT / "data/fixture_rounds/v59_20261108/manifest.json"),
        round_counts=mapping.provenance["counts"], team_directory_digest=directory["sha256"],
        supplied_teams=len(directory["teams"]), round_fixture_sides=122976,
        uniquely_resolved_round_fixture_sides=sides[1], canonical_teams_in_round_population=len(canonical),
        distinct_dated_canonical_fixture_keys=len(fixture_keys), self_matches=0, dated_fixture_key_collisions=0,
        unresolved_sides=0, ambiguous_sides=0, supplied_badge_formats=dict(formats),
        original_badge_entries_and_hashes_preserved=len(before), total_badges=len(current),
        canonical_ids_rewritten=False, workbook_ids_rewritten=False,
        interoperability_test_sha256_lf=hashlib.sha256((ROOT / "tests/analytics/test_round_team_compatibility.py").read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
        tests=tests, runtimes=json.loads((HERE / "runtimes.json").read_text(encoding="utf-8")),
        test_receipt_redaction="Hostname removed; outcomes and timings unchanged",
        experiment_prepared=False, experiment_run=False, forecast_timing_changed=False,
        publication_populations_expanded=False, new_worktrees_created=0)
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
