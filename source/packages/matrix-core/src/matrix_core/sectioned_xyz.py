from __future__ import annotations

from pathlib import Path
import re


_SECTION_HEADER_RE = re.compile(r"^\s*#[A-Za-z][A-Za-z0-9_]*\s*$")


def read_sectioned_lines(path: Path) -> list[str]:
    target = Path(path)
    if not target.exists():
        return []
    return target.read_text(encoding="utf-8").splitlines()


def write_sectioned_lines(path: Path, lines: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def normalize_section_name(section_name: str) -> str:
    text = section_name.strip().upper()
    if text.startswith("#"):
        text = text[1:].strip()
    if not text:
        raise ValueError("section name cannot be empty")
    return text


def section_header(section_name: str) -> str:
    return f"#{normalize_section_name(section_name)}"


def is_section_header_line(line: str) -> bool:
    return bool(_SECTION_HEADER_RE.match(line))


def section_content(lines: list[str], section_name: str) -> list[str]:
    header = section_header(section_name)
    start = None
    for idx, line in enumerate(lines):
        if line.strip().upper() == header:
            start = idx + 1
            break
    if start is None:
        return []
    end = len(lines)
    for idx in range(start, len(lines)):
        if is_section_header_line(lines[idx]):
            end = idx
            break
    return list(lines[start:end])


def has_section(path: Path, section_name: str) -> bool:
    header = section_header(section_name)
    return any(line.strip().upper() == header for line in read_sectioned_lines(path))


def remove_section_from_lines(lines: list[str], section_name: str) -> list[str]:
    header = section_header(section_name)
    out: list[str] = []
    skip = False
    for line in lines:
        if line.strip().upper() == header:
            skip = True
            continue
        if skip:
            if is_section_header_line(line):
                skip = False
                out.append(line)
            continue
        out.append(line)
    return out


def replace_section_in_lines(
    lines: list[str],
    section_name: str,
    content_lines: list[str],
) -> list[str]:
    header = section_header(section_name)
    for idx, line in enumerate(lines):
        if line.strip().upper() != header:
            continue
        end = len(lines)
        for end_idx in range(idx + 1, len(lines)):
            if is_section_header_line(lines[end_idx]):
                end = end_idx
                break
        return [*lines[: idx + 1], *content_lines, *lines[end:]]

    out = list(lines)
    while out and not out[-1].strip():
        out.pop()
    if out:
        out.append("")
    out.append(header)
    out.extend(content_lines)
    return out


def replace_section(path: Path, section_name: str, content_lines: list[str]) -> None:
    lines = read_sectioned_lines(path)
    write_sectioned_lines(path, replace_section_in_lines(lines, section_name, content_lines))


def xyz_tail_start(lines: list[str]) -> int:
    if not lines:
        return 0
    try:
        natoms = int(lines[0].strip())
    except ValueError:
        return 0
    return min(len(lines), natoms + 2)


def replace_xyz_block_in_lines(lines: list[str], xyz_lines: list[str]) -> list[str]:
    if len(xyz_lines) < 2:
        raise ValueError("XYZ block needs at least atom-count and comment lines")
    try:
        natoms = int(xyz_lines[0].strip())
    except ValueError as exc:
        raise ValueError("XYZ block first line must be an atom count") from exc
    if len(xyz_lines) < natoms + 2:
        raise ValueError("XYZ block is incomplete")
    return [*xyz_lines[: natoms + 2], *lines[xyz_tail_start(lines) :]]


def replace_xyz_block(path: Path, xyz_lines: list[str]) -> None:
    lines = read_sectioned_lines(path)
    write_sectioned_lines(path, replace_xyz_block_in_lines(lines, xyz_lines))
