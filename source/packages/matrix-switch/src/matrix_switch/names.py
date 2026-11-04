"""Auditable common-name to SMILES resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib.resources import files
import json
from pathlib import Path
import re
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from matrix_core import atomic_json_write


PUBCHEM_PUG_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_REMOTE_NAME_ALIASES = {
    "acido acetico": "acetic acid",
    "cisplatino": "cisplatin",
    "ferrocene": "ferrocene",
    "glucosio": "glucose",
    "urea": "urea",
}


class NameResolutionError(LookupError):
    pass


@dataclass(frozen=True)
class NameResolution:
    name: str
    smiles: str
    source: str
    identifier: str | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_name(
    name: str,
    *,
    allow_remote: bool = False,
    cache_path: Path | None = None,
    timeout: float = 10.0,
) -> NameResolution:
    query = str(name).strip()
    key = _normalize_name(query)
    if not key:
        raise NameResolutionError("molecule name is empty")
    resident = _resident_names()
    if key in resident:
        entry = resident[key]
        return NameResolution(
            name=query,
            smiles=str(entry["smiles"]),
            source="MATRIX_RESIDENT",
            identifier=str(entry.get("identifier") or key),
        )
    cache = _read_cache(cache_path)
    if key in cache:
        entry = cache[key]
        return NameResolution(
            name=query,
            smiles=str(entry["smiles"]),
            source=str(entry.get("source", "PUBCHEM_PUG_REST")),
            identifier=(
                str(entry["identifier"])
                if entry.get("identifier") is not None
                else (
                    f"PubChem CID {entry['cid']}"
                    if entry.get("cid") is not None
                    else None
                )
            ),
            cached=True,
        )
    if not allow_remote:
        raise NameResolutionError(
            f"{query!r} is not in the resident SWITCH name catalog; "
            "enable remote resolution explicitly"
        )
    result = _resolve_pubchem(_REMOTE_NAME_ALIASES.get(key, query), timeout=timeout)
    result = NameResolution(
        name=query,
        smiles=result.smiles,
        source=result.source,
        identifier=result.identifier,
    )
    target = _default_cache_path() if cache_path is None else Path(cache_path)
    cache[key] = {
        "smiles": result.smiles,
        "source": result.source,
        "identifier": result.identifier,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_cache(target, cache)
    return result


def _resolve_pubchem(name: str, *, timeout: float) -> NameResolution:
    endpoint = (
        f"{PUBCHEM_PUG_REST}/compound/name/{quote(name, safe='')}"
        "/property/SMILES/JSON"
    )
    request = Request(endpoint, headers={"User-Agent": "MATRIX-SWITCH/0.1"})
    try:
        with urlopen(request, timeout=float(timeout)) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise NameResolutionError(f"PubChem found no compound named {name!r}") from exc
        raise NameResolutionError(f"PubChem returned HTTP {exc.code} for {name!r}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise NameResolutionError(f"PubChem lookup failed for {name!r}: {exc}") from exc
    try:
        record = payload["PropertyTable"]["Properties"][0]
        smiles = next(
            str(record[field])
            for field in ("SMILES", "IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES")
            if record.get(field)
        )
    except (KeyError, IndexError, StopIteration, TypeError) as exc:
        raise NameResolutionError(f"PubChem returned no SMILES for {name!r}") from exc
    identifier = None if record.get("CID") is None else f"PubChem CID {record['CID']}"
    return NameResolution(
        name=name,
        smiles=smiles,
        source="PUBCHEM_PUG_REST",
        identifier=identifier,
    )


def _normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold()).strip()


@lru_cache(maxsize=1)
def _resident_names() -> dict[str, dict[str, str]]:
    resource = files("matrix_switch").joinpath("data/common_names.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return {_normalize_name(name): dict(entry) for name, entry in payload.items()}


def _default_cache_path() -> Path:
    return Path.home() / ".cache" / "matrix" / "switch_names.json"


def _read_cache(path: Path | None) -> dict[str, dict[str, object]]:
    target = _default_cache_path() if path is None else Path(path)
    if not target.is_file():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_cache(path: Path, cache: dict[str, dict[str, object]]) -> None:
    atomic_json_write(path, cache)


__all__ = [
    "NameResolution",
    "NameResolutionError",
    "PUBCHEM_PUG_REST",
    "resolve_name",
]
