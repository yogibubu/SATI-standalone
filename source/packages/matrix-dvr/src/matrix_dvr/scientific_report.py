"""Unified HTML/PDF report for SONIC/DVR scientific results."""

from __future__ import annotations

import html
import json
from pathlib import Path


def write_scientific_report(outdir: Path, *, result: dict[str, object], title: str = "MATRIX SONIC/DVR scientific report") -> dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    body = html.escape(json.dumps(result, indent=2, ensure_ascii=False))
    document = f"<!doctype html><html><meta charset='utf-8'><title>{html.escape(title)}</title><style>body{{font:15px system-ui;max-width:1100px;margin:2em auto}}pre{{background:#f4f4f4;padding:1em;white-space:pre-wrap}}</style><h1>{html.escape(title)}</h1><p>Surface, metric, rovibrational and uncertainty artifacts are listed below.</p><pre>{body}</pre></html>"
    html_path = outdir / "scientific_report.html"
    html_path.write_text(document, encoding="utf-8")
    pdf_path = outdir / "scientific_report.pdf"
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        doc.build([Paragraph(title, styles["Title"]), Spacer(1, 12), Preformatted(json.dumps(result, indent=2, ensure_ascii=False), styles["Code"])])
        return {"html": str(html_path), "pdf": str(pdf_path)}
    except ImportError:
        return {"html": str(html_path), "pdf": "unavailable: install reportlab"}


__all__ = ["write_scientific_report"]
