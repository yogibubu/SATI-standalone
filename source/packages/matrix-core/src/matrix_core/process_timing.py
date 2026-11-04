"""Measure complete child processes without timeout-polling latency."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
from threading import Lock, Timer
import time
from typing import IO, Mapping, Sequence


@dataclass(frozen=True)
class ProcessTiming:
    command: tuple[str, ...]
    returncode: int
    wall_seconds: float


def time_process(command: Sequence[str | os.PathLike], *, cwd: str | Path | None = None,
                 env: Mapping[str, str] | None = None, stdout: IO | int | None = None,
                 stderr: IO | int | None = subprocess.STDOUT,
                 timeout_seconds: float | None = None, check: bool = True) -> ProcessTiming:
    """Time spawn through exit using a blocking wait and a separate watchdog.

    No shell is used. Output may be inherited, redirected or discarded; PIPE
    is rejected because this function does not consume a pipe. The timeout
    kills and reaps the direct child, not arbitrary descendant processes.
    """
    args = tuple(os.fspath(value) for value in command)
    if not args or any(not value for value in args):
        raise ValueError('A nonempty executable argument vector is required')
    if stdout == subprocess.PIPE or stderr == subprocess.PIPE:
        raise ValueError('Redirect output to a file; PIPE is not consumed')
    if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError('timeout_seconds must be finite and positive')
    lock = Lock()
    expired = False
    started = time.perf_counter()
    process = subprocess.Popen(args, cwd=cwd, env=env, stdout=stdout, stderr=stderr)

    def expire():
        nonlocal expired
        with lock:
            if process.poll() is None:
                expired = True
                process.kill()

    watchdog = None
    if timeout_seconds is not None:
        watchdog = Timer(timeout_seconds, expire)
        watchdog.daemon = True
        watchdog.start()
    try:
        returncode = process.wait()
        elapsed = time.perf_counter() - started
    except BaseException:
        with lock:
            if process.poll() is None:
                process.kill()
        process.wait()
        raise
    finally:
        if watchdog is not None:
            watchdog.cancel()
            watchdog.join()
    if expired:
        raise subprocess.TimeoutExpired(args, timeout_seconds)
    if check and returncode:
        raise subprocess.CalledProcessError(returncode, args)
    return ProcessTiming(args, returncode, elapsed)
