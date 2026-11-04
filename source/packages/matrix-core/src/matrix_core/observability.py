"""Opt-in, tool-agnostic timing events for MATRIX operational paths."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from typing import IO, Iterator


@contextmanager
def observe_operation(
    operation: str,
    *,
    sink: IO[str] | None = None,
) -> Iterator[None]:
    """Emit one structured timing event when ``MATRIX_OBSERVABILITY_LOG`` is set.

    The context never changes the wrapped operation's exceptions or return
    values. The default path is silent, preserving all existing CLI output.
    """

    enabled = bool(os.environ.get("MATRIX_OBSERVABILITY_LOG", "").strip())
    started = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        if enabled:
            target = sink or sys.stderr
            payload = {
                "schema": "matrix.observability.event.v1",
                "operation": str(operation),
                "status": status,
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "pid": os.getpid(),
            }
            target.write(json.dumps(payload, sort_keys=True) + "\n")
            target.flush()


def append_event(path: Path | str, payload: dict[str, object]) -> Path:
    """Append a JSON-lines event for long-running external monitoring."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return target


__all__ = ["append_event", "observe_operation"]
