from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import io
import math

from matrix_qm import read_transitions_section
from matrix_thermo import read_thermo_section
from matrix_thermo.models import THERMO_KEYS, THERMO_LABELS, ThermoSection


@dataclass(frozen=True)
class PublicationExportResult:
    outdir: Path
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class SpectrumPeak:
    label: str
    position: float
    intensity: float


def export_thermo_table(
    xyzin: Path | str,
    outdir: Path | str,
    *,
    basename: str | None = None,
    formats: tuple[str, ...] = ("csv", "tex", "svg", "pdf"),
) -> PublicationExportResult:
    target = Path(xyzin)
    section = read_thermo_section(target)
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    stem = basename or f"{target.stem}.thermo"
    requested = tuple(_normalize_format(item) for item in formats)
    rows = _thermo_rows(section)
    paths: list[Path] = []
    for fmt in requested:
        if fmt == "csv":
            paths.append(_write_csv(destination / f"{stem}.csv", rows))
        elif fmt == "tex":
            paths.append(_write_latex(destination / f"{stem}.tex", rows))
        elif fmt == "svg":
            paths.append(_write_svg(destination / f"{stem}.svg", rows))
        elif fmt == "pdf":
            paths.append(_write_pdf(destination / f"{stem}.pdf", rows))
        else:
            raise ValueError(f"unsupported publication export format: {fmt}")
    return PublicationExportResult(destination, tuple(paths))


def export_electronic_spectrum(
    xyzin: Path | str,
    outdir: Path | str,
    *,
    basename: str | None = None,
    formats: tuple[str, ...] = ("csv", "svg", "pdf"),
    broadening_ev: float = 0.08,
) -> PublicationExportResult:
    target = Path(xyzin)
    transitions = read_transitions_section(target).transitions
    peaks = tuple(
        SpectrumPeak(
            label=f"{record.from_state}->{record.to_state}",
            position=record.energy_ev,
            intensity=(
                1.0
                if record.oscillator_strength is None or record.oscillator_strength <= 0.0
                else record.oscillator_strength
            ),
        )
        for record in transitions
    )
    return export_spectrum_publication(
        peaks,
        outdir,
        basename=basename or f"{target.stem}.electronic",
        formats=formats,
        x_label="Energy / eV",
        y_label="Oscillator strength",
        broadening=broadening_ev,
    )


def export_spectrum_publication(
    peaks: tuple[SpectrumPeak, ...],
    outdir: Path | str,
    *,
    basename: str,
    formats: tuple[str, ...] = ("csv", "svg", "pdf"),
    x_label: str = "Position",
    y_label: str = "Intensity",
    broadening: float = 0.08,
) -> PublicationExportResult:
    if not peaks:
        raise ValueError("cannot export an empty spectrum")
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    requested = tuple(_normalize_format(item) for item in formats)
    paths: list[Path] = []
    for fmt in requested:
        if fmt == "csv":
            paths.append(
                _write_spectrum_csv(destination / f"{basename}.csv", peaks, x_label, y_label)
            )
        elif fmt == "svg":
            paths.append(
                _write_spectrum_svg(
                    destination / f"{basename}.svg",
                    peaks,
                    x_label=x_label,
                    y_label=y_label,
                    broadening=broadening,
                )
            )
        elif fmt == "pdf":
            paths.append(
                _write_spectrum_pdf(
                    destination / f"{basename}.pdf",
                    peaks,
                    x_label=x_label,
                    y_label=y_label,
                )
            )
        else:
            raise ValueError(f"unsupported spectrum export format: {fmt}")
    return PublicationExportResult(destination, tuple(paths))


def _normalize_format(value: str) -> str:
    text = value.strip().lower().lstrip(".")
    if text == "latex":
        return "tex"
    return text


def _thermo_rows(section: ThermoSection) -> list[list[str]]:
    rows = [["component", *THERMO_KEYS]]
    for label in THERMO_LABELS:
        contribution = section.contribution(label)
        if contribution is None:
            continue
        rows.append(
            [
                label,
                *[_format_value(getattr(contribution, key)) for key in THERMO_KEYS],
            ]
        )
    return rows


def _format_value(value: float | None) -> str:
    return "" if value is None else f"{float(value):.12g}"


def _write_csv(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return path


def _write_latex(path: Path, rows: list[list[str]]) -> Path:
    column_spec = "l" + "r" * (len(rows[0]) - 1)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(_latex_cell(cell) for cell in rows[0]) + r" \\",
        r"\hline",
    ]
    for row in rows[1:]:
        lines.append(" & ".join(_latex_cell(cell) for cell in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_svg(path: Path, rows: list[list[str]]) -> Path:
    width = 1180
    line_height = 24
    height = 40 + line_height * len(rows)
    columns = len(rows[0])
    col_width = width / columns
    text = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Helvetica,Arial,sans-serif;font-size:13px}"
        ".head{font-weight:bold}</style>",
    ]
    for row_index, row in enumerate(rows):
        y = 28 + row_index * line_height
        cls = ' class="head"' if row_index == 0 else ""
        for column_index, cell in enumerate(row):
            x = 12 + column_index * col_width
            text.append(f'<text{cls} x="{x:.1f}" y="{y:.1f}">{_xml_escape(cell)}</text>')
    text.append("</svg>")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return path


def _write_pdf(path: Path, rows: list[list[str]]) -> Path:
    lines = ["  ".join(row) for row in rows]
    stream_lines = ["BT", "/F1 8 Tf", "40 780 Td"]
    for line in lines:
        stream_lines.append(f"({_pdf_escape(line)}) Tj")
        stream_lines.append("0 -14 Td")
    stream_lines.append("ET")
    stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(buffer.tell())
        buffer.write(f"{index} 0 obj\n".encode("ascii"))
        buffer.write(obj)
        buffer.write(b"\nendobj\n")
    xref = buffer.tell()
    buffer.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    buffer.write(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(buffer.getvalue())
    return path


def _write_spectrum_csv(
    path: Path,
    peaks: tuple[SpectrumPeak, ...],
    x_label: str,
    y_label: str,
) -> Path:
    rows = [["label", x_label, y_label]]
    for peak in peaks:
        rows.append([peak.label, _format_value(peak.position), _format_value(peak.intensity)])
    return _write_csv(path, rows)


def _write_spectrum_svg(
    path: Path,
    peaks: tuple[SpectrumPeak, ...],
    *,
    x_label: str,
    y_label: str,
    broadening: float,
) -> Path:
    width = 900
    height = 520
    left = 78
    right = 30
    top = 34
    bottom = 70
    xmin, xmax, ymax = _spectrum_bounds(peaks)
    envelope = _envelope_points(peaks, xmin=xmin, xmax=xmax, broadening=broadening, samples=360)

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value: float) -> float:
        return height - bottom - value / ymax * (height - top - bottom)

    envelope_points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in envelope)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        "<style>text{font-family:Helvetica,Arial,sans-serif;font-size:13px}"
        ".axis{stroke:#222;stroke-width:1.2}.stick{stroke:#1b6ca8;stroke-width:2}"
        ".env{fill:none;stroke:#d1495b;stroke-width:2}</style>",
        f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
        f'<polyline class="env" points="{envelope_points}"/>',
    ]
    for peak in peaks:
        x = sx(peak.position)
        lines.append(
            f'<line class="stick" x1="{x:.2f}" y1="{height - bottom:.2f}" '
            f'x2="{x:.2f}" y2="{sy(peak.intensity):.2f}"/>'
        )
        lines.append(
            f'<text x="{x + 4:.2f}" y="{sy(peak.intensity) - 5:.2f}">{_xml_escape(peak.label)}</text>'
        )
    lines.extend(
        [
            f'<text x="{width / 2:.1f}" y="{height - 24}">{_xml_escape(x_label)}</text>',
            f'<text x="16" y="{height / 2:.1f}" transform="rotate(-90 16 {height / 2:.1f})">{_xml_escape(y_label)}</text>',
            f'<text x="{left}" y="{height - bottom + 24}">{xmin:.3g}</text>',
            f'<text x="{width - right - 44}" y="{height - bottom + 24}">{xmax:.3g}</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_spectrum_pdf(
    path: Path,
    peaks: tuple[SpectrumPeak, ...],
    *,
    x_label: str,
    y_label: str,
) -> Path:
    width = 595
    height = 420
    left = 58
    right = 28
    bottom = 58
    top = 28
    xmin, xmax, ymax = _spectrum_bounds(peaks)

    def sx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * (width - left - right)

    def sy(value: float) -> float:
        return bottom + value / ymax * (height - top - bottom)

    stream_lines = [
        "1 w",
        f"{left} {bottom} m {width - right} {bottom} l S",
        f"{left} {bottom} m {left} {height - top} l S",
        "0.1 0.42 0.66 RG",
    ]
    for peak in peaks:
        x = sx(peak.position)
        stream_lines.append(f"{x:.2f} {bottom:.2f} m {x:.2f} {sy(peak.intensity):.2f} l S")
    stream_lines.extend(
        [
            "0 0 0 RG",
            "BT",
            "/F1 9 Tf",
            f"250 {bottom - 34} Td ({_pdf_escape(x_label)}) Tj",
            f"{left} {height - top + 12} Td ({_pdf_escape(y_label)}) Tj",
            "ET",
        ]
    )
    stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 420] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    _write_pdf_objects(path, objects)
    return path


def _write_pdf_objects(path: Path, objects: list[bytes]) -> None:
    buffer = io.BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(buffer.tell())
        buffer.write(f"{index} 0 obj\n".encode("ascii"))
        buffer.write(obj)
        buffer.write(b"\nendobj\n")
    xref = buffer.tell()
    buffer.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    buffer.write(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(buffer.getvalue())


def _spectrum_bounds(peaks: tuple[SpectrumPeak, ...]) -> tuple[float, float, float]:
    positions = [peak.position for peak in peaks]
    intensities = [max(0.0, peak.intensity) for peak in peaks]
    xmin = min(positions)
    xmax = max(positions)
    if math.isclose(xmin, xmax):
        xmin -= 0.5
        xmax += 0.5
    pad = 0.08 * (xmax - xmin)
    ymax = max(intensities) if intensities else 1.0
    return xmin - pad, xmax + pad, ymax * 1.15 if ymax > 0.0 else 1.0


def _envelope_points(
    peaks: tuple[SpectrumPeak, ...],
    *,
    xmin: float,
    xmax: float,
    broadening: float,
    samples: int,
) -> list[tuple[float, float]]:
    sigma = max(float(broadening), 1.0e-6)
    points: list[tuple[float, float]] = []
    for idx in range(samples):
        x = xmin + (xmax - xmin) * idx / (samples - 1)
        y = sum(
            max(0.0, peak.intensity) * math.exp(-0.5 * ((x - peak.position) / sigma) ** 2)
            for peak in peaks
        )
        points.append((x, y))
    return points


def _latex_cell(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
