import os

import pytest

from scraping.sofascore_web_v3.status_write import BACKOFF_SECONDS, atomic_write_status


def sharing_error(code=32):
    error = PermissionError("simulated Windows replacement conflict")
    error.winerror = code
    return error


def test_transient_replace_retries_same_complete_temporary_file(tmp_path, monkeypatch):
    path=tmp_path/"status.json"; path.write_bytes(b"old")
    real_replace=os.replace; attempts=[]; sleeps=[]
    def replace(source, destination):
        attempts.append(source)
        assert source.read_bytes()==b"new complete status"
        if len(attempts)<3:
            assert path.read_bytes()==b"old"
            raise sharing_error()
        real_replace(source,destination)
    monkeypatch.setattr(os,"replace",replace)
    atomic_write_status(path,b"new complete status",sleep=sleeps.append)
    assert len(set(attempts))==1 and len(attempts)==3
    assert sleeps==list(BACKOFF_SECONDS[:2])
    assert path.read_bytes()==b"new complete status"
    assert list(tmp_path.iterdir())==[path]


def test_persistent_conflict_preserves_status_and_cleans_own_temporary_file(tmp_path, monkeypatch):
    path=tmp_path/"status.json";path.write_bytes(b"old");sleeps=[]
    def replace(*args):
        raise sharing_error(5)
    monkeypatch.setattr(os,"replace",replace)
    with pytest.raises(PermissionError):
        atomic_write_status(path,b"new",sleep=sleeps.append)
    assert sleeps==list(BACKOFF_SECONDS) and path.read_bytes()==b"old"
    assert list(tmp_path.iterdir())==[path]


def test_unrelated_filesystem_error_is_not_retried(tmp_path, monkeypatch):
    path=tmp_path/"status.json";path.write_bytes(b"old");sleeps=[]
    def replace(*args):
        raise OSError(28,"disk full")
    monkeypatch.setattr(os,"replace",replace)
    with pytest.raises(OSError,match="disk full"):
        atomic_write_status(path,b"new",sleep=sleeps.append)
    assert sleeps==[] and path.read_bytes()==b"old"
    assert list(tmp_path.iterdir())==[path]


@pytest.mark.skipif(os.name!="nt",reason="Uses a real Windows reader sharing conflict")
def test_real_windows_reader_can_close_during_retry(tmp_path):
    path=tmp_path/"status.json";path.write_bytes(b"old")
    reader=path.open("rb");sleeps=[]
    def release(seconds):
        sleeps.append(seconds)
        assert reader.read()==b"old"
        reader.close()
    try:
        atomic_write_status(path,b"new",sleep=release)
    finally:
        reader.close()
    assert sleeps==[BACKOFF_SECONDS[0]] and path.read_bytes()==b"new"
