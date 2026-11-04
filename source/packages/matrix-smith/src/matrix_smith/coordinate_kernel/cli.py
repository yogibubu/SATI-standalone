"""Command dispatcher for SMITH coordinate-kernel utilities."""
from __future__ import annotations

import argparse
from importlib import import_module
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    commands = {
        "auto-report": "auto_report_pipeline",
        "fragments": "fragment_pipeline",
        "modify-geometry": "modify_geom",
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands)
    if not arguments or arguments[0] not in commands:
        parser.parse_args(arguments)
        return
    module = import_module(f"{__package__}.{commands[arguments[0]]}")
    if arguments[0] != "modify-geometry":
        return module.main(arguments[1:])
    saved = sys.argv
    try:
        sys.argv = [saved[0], *arguments[1:]]
        return module.main()
    finally:
        sys.argv = saved
