"""Dependency-free local dashboard for inspecting an ESALC result directory."""

from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def build_dashboard(outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    result = outdir / "esalc_scan_result.json"
    payload = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else {"status": "no result yet"}
    page = f"""<!doctype html><meta charset='utf-8'><title>MATRIX ESALC dashboard</title><style>body{{font:15px system-ui;max-width:1100px;margin:2em auto}}pre{{background:#f4f4f4;padding:1em;overflow:auto}}iframe{{width:100%;height:650px;border:1px solid #aaa}}</style><h1>MATRIX ESALC dashboard</h1><p>Refresh the page to inspect newly completed artifacts.</p><pre>{json.dumps(payload,indent=2)}</pre><iframe src='esalc_scan_viewer.html'></iframe>"""
    path = outdir / "dashboard.html"
    path.write_text(page, encoding="utf-8")
    return path


def serve_dashboard(outdir: Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    directory = str(outdir.resolve())
    handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=directory, **kwargs)
    with ThreadingHTTPServer((host, int(port)), handler) as server:
        server.serve_forever()


__all__ = ["build_dashboard", "serve_dashboard"]
