"""Atomic status replacement with bounded retries for Windows sharing conflicts."""
import os
from pathlib import Path
import tempfile
import time

BACKOFF_SECONDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8)


def atomic_write_status(path, data, *, sleep=time.sleep):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(len(BACKOFF_SECONDS) + 1):
            try:
                os.replace(temporary, path)
                return
            except OSError as error:
                if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == len(BACKOFF_SECONDS):
                    raise
                sleep(BACKOFF_SECONDS[attempt])
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # Preserve the original error if cleanup is also locked.
