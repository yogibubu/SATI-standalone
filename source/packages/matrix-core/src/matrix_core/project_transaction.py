"""Transactional publication of complete MATRIX project directories."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

from .atomic_io import atomic_json_write


PROJECT_LIFECYCLE_SCHEMA = "matrix.project_lifecycle.v1"
PROJECT_LIFECYCLE_FILENAME = "state/project-lifecycle.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProjectTransaction:
    """Build a new project privately and publish it atomically on commit.

    A transaction never adopts or overwrites an existing destination.  Closing
    an uncommitted transaction removes only its UUID-qualified sibling staging
    directory; a committed project is retained by explicit policy.
    """

    destination: Path
    preservation: str = "retain"
    transaction_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    _staging: Path | None = field(default=None, init=False, repr=False)
    _committed: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.destination = Path(self.destination).expanduser().resolve()
        if self.preservation != "retain":
            raise ValueError("MATRIX project preservation policy must be 'retain'")

    @property
    def root(self) -> Path:
        if self._staging is None:
            raise RuntimeError("project transaction has not been opened")
        return self._staging

    @property
    def committed(self) -> bool:
        return self._committed

    def __enter__(self) -> "ProjectTransaction":
        if self._staging is not None or self._closed:
            raise RuntimeError("project transaction cannot be reopened")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        if self.destination.exists():
            raise FileExistsError(self.destination)
        staging = self.destination.parent / (
            f".{self.destination.name}.matrix-staging-{self.transaction_id}"
        )
        staging.mkdir(parents=False, exist_ok=False)
        self._staging = staging
        return self

    def commit(self) -> Path:
        if self._staging is None or self._closed:
            raise RuntimeError("project transaction is not open")
        if self._committed:
            return self.destination
        if self.destination.exists():
            raise FileExistsError(self.destination)
        lifecycle = self._staging / PROJECT_LIFECYCLE_FILENAME
        lifecycle.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": PROJECT_LIFECYCLE_SCHEMA,
            "transaction_id": self.transaction_id,
            "status": "committed",
            "committed_utc": _utc_now(),
            "destination_name": self.destination.name,
            "preservation": {
                "project": self.preservation,
                "temporary_artifacts": "remove_on_close",
            },
        }
        atomic_json_write(lifecycle, payload)
        self._staging.replace(self.destination)
        self._committed = True
        return self.destination

    def close(self) -> None:
        if self._closed:
            return
        if not self._committed and self._staging is not None and self._staging.exists():
            expected_prefix = f".{self.destination.name}.matrix-staging-"
            if (
                self._staging.parent.resolve() != self.destination.parent.resolve()
                or not self._staging.name.startswith(expected_prefix)
                or self.transaction_id not in self._staging.name
            ):
                raise RuntimeError("refusing to clean an unrecognised project staging path")
            shutil.rmtree(self._staging)
        self._closed = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
