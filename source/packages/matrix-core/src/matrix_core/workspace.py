from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import uuid


WORKSPACE_DIRS = ("inputs", "runs", "outputs", "reports", "cache", "logs", "state")


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return text or "run"


@dataclass(frozen=True)
class WorkspaceLayout:
    root: Path

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def state(self) -> Path:
        """Persistent orchestration state, separate from scientific results."""
        return self.root / "state"

    def ensure(self) -> "WorkspaceLayout":
        self.root.mkdir(parents=True, exist_ok=True)
        for name in WORKSPACE_DIRS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self

    def new_run_dir(self, workflow: str) -> Path:
        self.ensure()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.runs / f"{stamp}-{slugify(workflow)}-{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path


def ensure_workspace(root: Path) -> WorkspaceLayout:
    return WorkspaceLayout(Path(root)).ensure()
