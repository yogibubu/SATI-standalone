"""Standalone launcher for the canonical MATRIX ORACLE window."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable


@dataclass(frozen=True)
class _GuiLaunchOptions:
    """Pure launch description kept separate from the optional Qt import."""

    xyzin: Path | None
    workdir: Path | None
    smoke_test: bool

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> _GuiLaunchOptions:
        return cls(
            xyzin=args.xyzin,
            workdir=args.workdir,
            smoke_test=args.smoke_test,
        )

    def forwarded_arguments(self) -> list[str]:
        forwarded = ["ORACLE"]
        if self.xyzin is not None:
            forwarded.append(str(self.xyzin))
        if self.workdir is not None:
            forwarded.extend(("--workdir", str(self.workdir)))
        if self.smoke_test:
            forwarded.append("--smoke-test")
        return forwarded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oracle-gui",
        description="Open the canonical MATRIX ORACLE desktop window",
    )
    parser.add_argument(
        "xyzin",
        nargs="?",
        type=Path,
        help="Existing ORACLE enriched XYZ project",
    )
    parser.add_argument("--workdir", type=Path, help="Workspace for runs and reports")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Create the offscreen window and exit immediately",
    )
    return parser


def run(argv: list[str] | None = None) -> int:
    options = _GuiLaunchOptions.from_namespace(build_parser().parse_args(argv))
    runner = _load_matrix_gui_runner()
    return runner(options.forwarded_arguments())


def _load_matrix_gui_runner() -> Callable[[list[str] | None], int]:
    """Load the GUI only at the optional-dependency boundary."""

    try:
        from matrix_gui.app import run as run_matrix_gui
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith(("matrix_gui", "PySide6")):
            raise SystemExit(
                "The ORACLE desktop requires the GUI extra: "
                "`python -m pip install matrix-oracle[qt]`."
            ) from exc
        raise
    return run_matrix_gui


def main() -> None:
    raise SystemExit(run(sys.argv[1:]))


__all__ = ["build_parser", "main", "run"]
