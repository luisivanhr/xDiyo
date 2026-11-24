"""Saved batch execution must not prepare or change historical selections."""
import json
import sqlite3
import sys

import pytest

from scraping import run_archive_expansion as runner
from scraping.sofascore_web_v3.archive_discovery import ArchiveDiscovery, prepare_archive
from scraping.sofascore_web_v3.archive_formats import _spec


def saved_batch(tmp_path):
    return prepare_archive("new-league", [_spec("New_League", 325, "2017")],
                           base=tmp_path / "batches", discovery_base=tmp_path / "discovery",
                           output=tmp_path / "output")


def test_watch_saved_batch_does_not_prepare_old_groups(tmp_path, monkeypatch):
    batch = saved_batch(tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "--watch-batch", batch["path"]])
    monkeypatch.setattr(runner, "prepare_selected", lambda: pytest.fail("Touched old selections"))
    watched = []
    monkeypatch.setattr(runner, "watch_archive", watched.append)
    runner.main()
    assert [b["discovery_id"] for b in watched] == [batch["discovery_id"]]


def test_changed_saved_batch_is_rejected_before_watching(tmp_path, monkeypatch):
    batch = saved_batch(tmp_path)
    batch["settings"]["output_root"] = str(tmp_path / "different-output")
    from pathlib import Path
    Path(batch["path"]).write_text(json.dumps(batch), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["runner", "--watch-batch", batch["path"]])
    monkeypatch.setattr(runner, "watch_archive", lambda _: pytest.fail("Watched conflicting batch"))
    with pytest.raises(ValueError, match="differs"):
        runner.main()


def test_saved_batch_status_does_not_start_watcher(tmp_path, monkeypatch, capsys):
    batch = saved_batch(tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "--watch-batch", batch["path"], "--status"])
    monkeypatch.setattr(runner, "prepare_selected", lambda: pytest.fail("Touched old selections"))
    monkeypatch.setattr(runner, "watch_archive", lambda _: pytest.fail("Started a watcher"))
    runner.main()
    assert json.loads(capsys.readouterr().out)["discovery_id"] == batch["discovery_id"]


def test_incomplete_watch_polls_progress_without_revalidating_seasons(tmp_path, monkeypatch, capsys):
    batch = saved_batch(tmp_path)
    monkeypatch.setattr(ArchiveDiscovery, "status", lambda *a, **k: pytest.fail("Revalidated incomplete discovery"))
    polls = []

    def advance(seconds):
        polls.append(seconds)
        if len(polls) == 2:
            raise KeyboardInterrupt
        with sqlite3.connect(str(tmp_path / "discovery" / batch["discovery_id"] / "discovery.sqlite")) as db:
            db.execute("UPDATE jobs SET state='running'")

    with pytest.raises(KeyboardInterrupt):
        runner.watch_archive(batch, sleep=advance)
    output = capsys.readouterr().out
    assert "{'pending': 1}" in output and "{'running': 1}" in output
    assert polls == [5, 5]


def test_finished_watch_revalidates_changed_data_even_when_job_counts_match(tmp_path, monkeypatch):
    batch = saved_batch(tmp_path)
    database = tmp_path / "discovery" / batch["discovery_id"] / "discovery.sqlite"
    with sqlite3.connect(str(database)) as db:
        db.execute("UPDATE jobs SET state='complete'")
    validate = ArchiveDiscovery.status
    checked = []

    def record(discovery, **kwargs):
        result = validate(discovery, **kwargs)
        checked.append(result)
        return result

    monkeypatch.setattr(ArchiveDiscovery, "status", record)
    polls = []

    def advance(seconds):
        polls.append(seconds)
        if len(polls) == 3:
            raise KeyboardInterrupt
        if len(polls) == 2:
            with sqlite3.connect(str(database)) as db:
                db.execute("INSERT OR REPLACE INTO state VALUES('missing:0',?)", (json.dumps("New provider identity issue"),))

    with pytest.raises(KeyboardInterrupt):
        runner.watch_archive(batch, sleep=advance)
    assert len(checked) == 2
    assert checked[0]["jobs"] == checked[1]["jobs"] == {"complete": 1}
    assert "New provider identity issue" not in checked[0]["seasons"][0]["validation_issues"]
    assert "New provider identity issue" in checked[1]["seasons"][0]["validation_issues"]
