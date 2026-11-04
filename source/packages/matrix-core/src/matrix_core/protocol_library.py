"""Installed, immutable KEYMAKER protocol library.

Calculation protocol classes remain owner-neutral in ``calculation_protocols``.
This module binds exact KEYMAKER actions to versioned owner implementations and
their typed artifact gates.  Protocol resources are package data, so dispatch
never depends on a source checkout or a repository-relative path.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import resources
import json
import re
from typing import Any, Mapping

from .calculation_protocols import load_calculation_protocol_atlas
from .operation_registry import operation_contract


KEYMAKER_PROTOCOL_MANIFEST_SCHEMA = "matrix.keymaker.protocol_manifest.v1"
KEYMAKER_PROTOCOL_ENTRYPOINT = "KEYMAKER"
KEYMAKER_PROTOCOL_CHANGE_POLICY = "new_version_and_explicit_approval"
ACTIVE_GFN2_ZAFF_FIT_PROTOCOL_ID = "zaff-gfn2-residual-fit.v4"
GFN2_ZAFF_FIT_PROTOCOL_IDS = frozenset({
    "zaff-gfn2-residual-fit.v1",
    "zaff-gfn2-residual-fit.v2",
    "zaff-gfn2-residual-fit.v3",
    ACTIVE_GFN2_ZAFF_FIT_PROTOCOL_ID,
})
TWO_MORSE_GFN2_ZAFF_FIT_PROTOCOL_IDS = frozenset({
    "zaff-gfn2-residual-fit.v3", ACTIVE_GFN2_ZAFF_FIT_PROTOCOL_ID,
})
EXPECTED_KEYMAKER_PROTOCOL_LIBRARY_SHA256 = (
    "b2757598fde04cc569ef9f002a8fdd587df51e0238f185549f7f61ffe2ac9560"
)
EXPECTED_KEYMAKER_PROTOCOL_CATALOG_SHA256 = "6916a577fa935d3f9dd5e2e0a78c8584d5b36a356cb8ebfc1eac1ebf82535beb"
KNOWN_KEYMAKER_PROTOCOL_LIBRARY_SHA256S = frozenset({
    "83273228da60fb717b7213cebf31dca321e9cefdf7caaba8bc13975ae500aa8f",
    "db3c59bed663f253f77eeab851475ec96a1fdd3cef963038d7e9ed39b84a6f94",
    "2609c2d03e7544618fbbde1da7937f3cf8c6b19f4946054541c51dc3815432e0",
    "3ca65b96e1043ea7a7c43fedcb2da4c6f2b6f57a9c250906bc4c71e6b9b230fc",
    "b017460456a0df9fa41956bad4ad9ccfef3ae2b5846ee8c0abbba4593f35ba48",
    "580a54494e9b230f1b7be000f180ddeaecba0e16009f1567f6ae0bafdca440a0",
    "4c8aa1174e5259513f6d874f8d25da969f8321f572c6534d06b6ccd423c836fa",
    "6da81e5be6425aa0eedf6c22648e32e1c455315c9fe2cbc3a590f977f1242a54",
    "c456ccc64914e05c04dcb32845ae6819179ebb053c63a54c8165c5e81bcf73a6",
    "76e4735f099557ee408ab519d4790fbe8d2c65b2b88f8dc52336423cf31e223c",
    "180f2547d3150e71484dabc78bd2b744466a3c013d4d336bb1f87bc195e9855c",
    "50f7ad8d8d60d59e77b206f913dfefb7e4a0f217c034589fa579978b4688fc75",
    "7d32d6459ca90c999b6fd925d5cb7750ef459b5e0c5058a81c08321cd30e1990",
    "ba24dfb4c02527ae049e572e65ff1887ff703187055367ccf3f92f565d8a2666",
    "ceb1273e2cd5b30b8ffda8de670c0e79735bfd6b749f948b12ca82f75ec39a17",
    "ead07fd3ec2d540468271c600a1088c887a012ab6633485c736662f726e43965",
    "43a8477c1034a7d35c2ac0494f7205c48a7e3a19085cc3ff5b01d53ca3cdd25e",
    "922c2e6866e31951729dac5a0374ba44e883434ad65ec8b2f37168066e7297ca",
    EXPECTED_KEYMAKER_PROTOCOL_LIBRARY_SHA256,
})
_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_REQUIRED_FIELDS = {
    "schema",
    "protocol_id",
    "manifest_version",
    "status",
    "owner_operation",
    "entrypoint",
    "title",
    "summary",
    "calculation_protocol",
    "aliases",
    "dependencies",
    "routes",
    "inputs",
    "outputs",
    "gates",
    "implementation",
    "coordinate_policy",
    "backend_policy",
    "change_policy",
}


def _token(value: object) -> str:
    return str(value).strip().casefold().replace("-", "_")


@dataclass(frozen=True)
class ProtocolArtifact:
    name: str
    schema: str
    required: bool

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProtocolArtifact":
        if set(payload) != {"name", "schema", "required"}:
            raise RuntimeError("KEYMAKER protocol artifact has incomplete fields")
        if not isinstance(payload["required"], bool):
            raise RuntimeError("KEYMAKER protocol artifact required must be a boolean")
        if any(not isinstance(payload[name], str) for name in ("name", "schema")):
            raise RuntimeError("KEYMAKER protocol artifact names and schemas must be strings")
        artifact = cls(
            name=str(payload["name"]).strip(),
            schema=str(payload["schema"]).strip(),
            required=bool(payload["required"]),
        )
        if not artifact.name or not artifact.schema:
            raise RuntimeError("KEYMAKER protocol artifact is incomplete")
        return artifact


@dataclass(frozen=True)
class ProtocolRoute:
    menu: str
    action: str
    option_equals: Mapping[str, str]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProtocolRoute":
        if set(payload) != {"menu", "action", "option_equals"}:
            raise RuntimeError("KEYMAKER exact protocol route has incomplete fields")
        raw_options = payload["option_equals"]
        if not isinstance(raw_options, Mapping):
            raise RuntimeError("KEYMAKER exact protocol route options must be an object")
        if any(not isinstance(payload[field], str) for field in ("menu", "action")):
            raise RuntimeError("KEYMAKER exact protocol route names must be strings")
        if any(not isinstance(key, str) or not key.strip() or not isinstance(value, str)
               or not value.strip() for key, value in raw_options.items()):
            raise RuntimeError("KEYMAKER exact protocol route options must be non-empty strings")
        route = cls(
            menu=str(payload["menu"]).strip(),
            action=str(payload["action"]).strip(),
            option_equals={str(key): str(value) for key, value in raw_options.items()},
        )
        if not route.menu or not route.action:
            raise RuntimeError("KEYMAKER exact protocol route is incomplete")
        return route


@dataclass(frozen=True)
class KeymakerProtocolManifest:
    protocol_id: str
    manifest_version: str
    status: str
    owner_operation: str
    title: str
    summary: str
    calculation_protocol: str
    aliases: tuple[str, ...]
    dependencies: tuple[str, ...]
    routes: tuple[ProtocolRoute, ...]
    inputs: tuple[ProtocolArtifact, ...]
    outputs: tuple[ProtocolArtifact, ...]
    gates: tuple[str, ...]
    owner_api: str
    action: str
    coordinate_policy: str
    backend_policy: str
    sha256: str
    payload: Mapping[str, Any]

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, sha256: str
    ) -> "KeymakerProtocolManifest":
        validate_keymaker_protocol_manifest(payload)
        implementation = payload["implementation"]
        return cls(
            protocol_id=str(payload["protocol_id"]),
            manifest_version=str(payload["manifest_version"]),
            status=str(payload["status"]),
            owner_operation=str(payload["owner_operation"]),
            title=str(payload["title"]),
            summary=str(payload["summary"]),
            calculation_protocol=str(payload["calculation_protocol"]),
            aliases=tuple(str(item) for item in payload["aliases"]),
            dependencies=tuple(str(item) for item in payload["dependencies"]),
            routes=tuple(ProtocolRoute.from_dict(item) for item in payload["routes"]),
            inputs=tuple(ProtocolArtifact.from_dict(item) for item in payload["inputs"]),
            outputs=tuple(ProtocolArtifact.from_dict(item) for item in payload["outputs"]),
            gates=tuple(str(item) for item in payload["gates"]),
            owner_api=str(implementation["owner_api"]),
            action=str(implementation["action"]),
            coordinate_policy=str(payload["coordinate_policy"]),
            backend_policy=str(payload["backend_policy"]),
            sha256=sha256,
            payload=deepcopy(dict(payload)),
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.payload))


@dataclass(frozen=True)
class KeymakerProtocolLibrary:
    manifests: tuple[KeymakerProtocolManifest, ...]
    sha256: str

    def list_protocols(self, *, status: str | None = None) -> tuple[KeymakerProtocolManifest, ...]:
        if status is None:
            return self.manifests
        return tuple(item for item in self.manifests if item.status == status)

    def resolve(self, identifier: str) -> KeymakerProtocolManifest:
        normalized = _token(identifier)
        matches = tuple(
            item
            for item in self.manifests
            if normalized in {_token(item.protocol_id), *(_token(alias) for alias in item.aliases)}
        )
        if len(matches) != 1:
            raise KeyError(f"unknown or ambiguous KEYMAKER protocol: {identifier}")
        return matches[0]

    def resolve_action(
        self,
        menu: str,
        action: str,
        *,
        options: Mapping[str, object] | None = None,
    ) -> KeymakerProtocolManifest:
        normalized_options = {
            str(key): _token(value) for key, value in dict(options or {}).items()
        }
        matches: list[KeymakerProtocolManifest] = []
        for manifest in self.manifests:
            for route in manifest.routes:
                if route.menu != str(menu) or route.action != str(action):
                    continue
                if all(
                    normalized_options.get(key) == _token(value)
                    for key, value in route.option_equals.items()
                ):
                    matches.append(manifest)
        if len(matches) != 1:
            raise KeyError(f"unknown or ambiguous KEYMAKER protocol route: {menu}/{action}")
        return matches[0]

    def dependency_order(self, identifier: str) -> tuple[KeymakerProtocolManifest, ...]:
        root = self.resolve(identifier)
        ordered: list[KeymakerProtocolManifest] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(item: KeymakerProtocolManifest) -> None:
            key = _token(item.protocol_id)
            if key in visiting:
                raise RuntimeError(f"cyclic KEYMAKER protocol dependency at {item.protocol_id}")
            if key in visited:
                return
            visiting.add(key)
            for dependency in item.dependencies:
                visit(self.resolve(dependency))
            visiting.remove(key)
            visited.add(key)
            ordered.append(item)

        visit(root)
        return tuple(ordered)

    def explain_route(
        self, menu: str, action: str, *, options: Mapping[str, object] | None = None
    ) -> dict[str, Any]:
        manifest = self.resolve_action(menu, action, options=options)
        return {
            "protocol_id": manifest.protocol_id,
            "manifest_version": manifest.manifest_version,
            "sha256": manifest.sha256,
            "owner_operation": manifest.owner_operation,
            "owner_api": manifest.owner_api,
            "action": manifest.action,
            "dependencies": [item.protocol_id for item in self.dependency_order(manifest.protocol_id)],
        }


def validate_keymaker_protocol_manifest(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _REQUIRED_FIELDS:
        raise RuntimeError("KEYMAKER protocol manifest has incomplete fields")
    array_fields = {"aliases", "dependencies", "routes", "inputs", "outputs", "gates"}
    if any(not isinstance(payload[field], str)
           for field in _REQUIRED_FIELDS - array_fields - {"implementation"}):
        raise RuntimeError("KEYMAKER protocol scalar fields must be strings")
    if payload.get("schema") != KEYMAKER_PROTOCOL_MANIFEST_SCHEMA:
        raise RuntimeError("unsupported KEYMAKER protocol manifest schema")
    if payload.get("entrypoint") != KEYMAKER_PROTOCOL_ENTRYPOINT:
        raise RuntimeError("KEYMAKER protocol has a non-KEYMAKER entrypoint")
    if payload.get("change_policy") != KEYMAKER_PROTOCOL_CHANGE_POLICY:
        raise RuntimeError("KEYMAKER protocol change policy was weakened")
    if payload.get("status") not in {"draft", "approved", "deprecated"}:
        raise RuntimeError("unsupported KEYMAKER protocol status")
    if not _VERSION.fullmatch(str(payload.get("manifest_version", ""))):
        raise RuntimeError("invalid KEYMAKER protocol manifest version")
    protocol_id = str(payload.get("protocol_id", "")).strip()
    if not protocol_id:
        raise RuntimeError("KEYMAKER protocol identifier is empty")
    owner = operation_contract(str(payload.get("owner_operation", "")))
    calculation_protocol = str(payload.get("calculation_protocol", ""))
    load_calculation_protocol_atlas().resolve(calculation_protocol)
    implementation = payload.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {"owner_api", "action"}:
        raise RuntimeError("KEYMAKER protocol implementation is incomplete")
    owner_api = str(implementation["owner_api"]).strip()
    if owner_api.split(".", 1)[0] != owner.owner_api.split(".", 1)[0]:
        raise RuntimeError("KEYMAKER protocol implementation escapes its scientific owner")
    if not str(implementation["action"]).strip():
        raise RuntimeError("KEYMAKER protocol implementation action is empty")
    for field in ("title", "summary", "coordinate_policy", "backend_policy"):
        if not str(payload.get(field, "")).strip():
            raise RuntimeError(f"KEYMAKER protocol field is empty: {field}")
    for field in ("aliases", "dependencies", "routes", "inputs", "outputs", "gates"):
        if not isinstance(payload.get(field), list):
            raise RuntimeError(f"KEYMAKER protocol field must be an array: {field}")
    for field in ("aliases", "dependencies", "gates"):
        if any(not isinstance(item, str) or not item.strip() for item in payload[field]):
            raise RuntimeError(f"KEYMAKER protocol {field} must contain non-empty strings")
    if not payload["gates"] or any(not str(item).strip() for item in payload["gates"]):
        raise RuntimeError("KEYMAKER protocol requires non-empty validation gates")
    aliases = [_token(item) for item in payload["aliases"]]
    dependencies = [_token(item) for item in payload["dependencies"]]
    if len(aliases) != len(set(aliases)) or len(dependencies) != len(set(dependencies)):
        raise RuntimeError("KEYMAKER protocol aliases and dependencies must be unique")
    for route in payload["routes"]:
        ProtocolRoute.from_dict(route)
    for collection in (payload["inputs"], payload["outputs"]):
        names: set[str] = set()
        for artifact in collection:
            parsed = ProtocolArtifact.from_dict(artifact)
            if parsed.name in names:
                raise RuntimeError("duplicate KEYMAKER protocol artifact name")
            names.add(parsed.name)


def validate_keymaker_protocol_library(library: KeymakerProtocolLibrary) -> None:
    """Validate ownership, dependencies and unambiguous routes as one contract."""
    if not library.manifests:
        raise RuntimeError("installed KEYMAKER protocol library is empty")
    atlas = load_calculation_protocol_atlas()
    identifiers: set[str] = set()
    routes: dict[tuple[str, str], list[ProtocolRoute]] = {}
    for manifest in library.manifests:
        validate_keymaker_protocol_manifest(manifest.payload)
        for value in (manifest.protocol_id, *manifest.aliases):
            normalized = _token(value)
            if not normalized or normalized in identifiers:
                raise RuntimeError("duplicate KEYMAKER protocol identifier or alias")
            identifiers.add(normalized)
        for route in manifest.routes:
            key = (route.menu, route.action)
            options = {name: _token(value) for name, value in route.option_equals.items()}
            for other in routes.get(key, []):
                other_options = {
                    name: _token(value) for name, value in other.option_equals.items()
                }
                shared = options.keys() & other_options.keys()
                if all(options[name] == other_options[name] for name in shared):
                    raise RuntimeError(
                        f"overlapping KEYMAKER protocol routes: {route.menu}/{route.action}"
                    )
            routes.setdefault(key, []).append(route)
            try:
                calculation = atlas.resolve_keymaker_action(
                    route.menu, route.action, options=route.option_equals
                )
            except KeyError as exc:
                raise RuntimeError(
                    f"KEYMAKER protocol route absent or ambiguous in atlas: "
                    f"{manifest.protocol_id}/{route.action}"
                ) from exc
            if calculation.protocol_id != manifest.calculation_protocol:
                raise RuntimeError(
                    f"KEYMAKER protocol disagrees with calculation atlas: {manifest.protocol_id}"
                )
    for manifest in library.manifests:
        library.dependency_order(manifest.protocol_id)


def load_keymaker_protocol_library() -> KeymakerProtocolLibrary:
    root = resources.files("matrix_core").joinpath("data/keymaker_protocols")
    entries = tuple(sorted((item for item in root.iterdir() if item.name.endswith(".json")), key=lambda item: item.name))
    if not entries:
        raise RuntimeError("installed KEYMAKER protocol library is empty")
    manifests: list[KeymakerProtocolManifest] = []
    library_hash = hashlib.sha256()
    for entry in entries:
        raw = entry.read_bytes()
        library_hash.update(entry.name.encode("utf-8"))
        library_hash.update(b"\0")
        library_hash.update(raw)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"unreadable KEYMAKER protocol manifest: {entry.name}") from exc
        manifests.append(
            KeymakerProtocolManifest.from_payload(
                payload, sha256=hashlib.sha256(raw).hexdigest()
            )
        )
    digest = library_hash.hexdigest()
    if digest != EXPECTED_KEYMAKER_PROTOCOL_LIBRARY_SHA256:
        raise RuntimeError("installed KEYMAKER protocol library differs from its frozen fingerprint")
    library = KeymakerProtocolLibrary(tuple(manifests), digest)
    validate_keymaker_protocol_library(library)
    _validate_catalog_index(library)
    return library


def _validate_catalog_index(library: KeymakerProtocolLibrary) -> dict[str, Any]:
    """Bind the installed manifest set and calculation atlas to one frozen index."""
    raw = resources.files("matrix_core").joinpath(
        "data/keymaker_protocol_catalog_v14.json"
    ).read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_KEYMAKER_PROTOCOL_CATALOG_SHA256:
        raise RuntimeError("installed KEYMAKER protocol catalog differs from its frozen fingerprint")
    index = json.loads(raw)
    expected = [
        {"protocol_id": item.protocol_id, "resource": item.protocol_id + ".json", "sha256": item.sha256}
        for item in library.manifests
    ]
    atlas = load_calculation_protocol_atlas()
    if (
        index.get("schema") != "matrix.keymaker.protocol_catalog.v1"
        or index.get("library_sha256") != library.sha256
        or index.get("protocols") != expected
        or index.get("calculation_atlas", {}).get("sha256") != atlas.sha256
        or index.get("calculation_atlas", {}).get("atlas_id") != atlas.payload["atlas_id"]
    ):
        raise RuntimeError("KEYMAKER protocol catalog membership or calculation atlas differs")
    return index


def keymaker_protocol_catalog(*, verify_implementations: bool = False) -> dict[str, Any]:
    """Return the single installed catalog and its explicit contract validation scope.

    Importing an implementation verifies availability only. It does not launch
    calculations or certify numerical or experimental accuracy.
    """
    from importlib import import_module

    library = load_keymaker_protocol_library()
    atlas = load_calculation_protocol_atlas()
    issues: list[str] = []
    implementations: list[dict[str, object]] = []
    declared = set()
    for manifest in library.manifests:
        for route in manifest.routes:
            declared.add((route.menu, route.action, tuple(sorted(
                (name, _token(value)) for name, value in route.option_equals.items()
            ))))
        if verify_implementations:
            try:
                module = import_module(manifest.owner_api)
                implementation = getattr(module, manifest.action)
                if not callable(implementation):
                    raise TypeError("declared implementation is not callable")
                implementations.append({
                    "protocol_id": manifest.protocol_id,
                    "owner_api": manifest.owner_api,
                    "action": manifest.action,
                    "module_path": str(getattr(module, "__file__", "")),
                })
            except (ImportError, AttributeError, TypeError, OSError) as exc:
                issues.append(f"{manifest.protocol_id}: {type(exc).__name__}: {exc}")
    unbound = [{"menu": route.menu, "action": route.action,
                "option_equals": dict(route.option_equals), "protocol": route.protocol_id}
               for route in atlas.keymaker_routes if (
        route.menu, route.action, tuple(sorted(
            (name, _token(value)) for name, value in route.option_equals.items()
        ))
    ) not in declared]
    return {
        "schema": "matrix.keymaker.protocol_catalog_audit.v1",
        "catalog": _validate_catalog_index(library),
        "validation_scope": "manifest hashes, ownership, route uniqueness, dependencies and atlas consistency",
        "protocol_count": len(library.manifests),
        "registered_route_count": len(declared),
        "atlas_route_count": len(atlas.keymaker_routes),
        "contracts_valid": not issues,
        "coverage_complete": not unbound,
        "implementations_verified": verify_implementations,
        "implementations": implementations,
        "issues": issues,
        "atlas_routes_without_manifest": unbound,
        "protocols": [manifest.to_dict() for manifest in library.manifests],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Validate and export the installed KEYMAKER protocol library.")
    parser.add_argument("--verify-implementations", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = keymaker_protocol_catalog(verify_implementations=args.verify_implementations)
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        result = {"contracts_valid": False, "coverage_complete": False, "issues": [str(exc)]}
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {"protocols", "catalog", "implementations"}}, indent=2))
    else:
        print(text, end="")
    return 0 if result["contracts_valid"] and (
        not args.require_complete or result["coverage_complete"]
    ) else 1


def resolve_keymaker_protocol(identifier: str) -> KeymakerProtocolManifest:
    return load_keymaker_protocol_library().resolve(identifier)


__all__ = [
    "KEYMAKER_PROTOCOL_CHANGE_POLICY",
    "KEYMAKER_PROTOCOL_ENTRYPOINT",
    "KEYMAKER_PROTOCOL_MANIFEST_SCHEMA",
    "KeymakerProtocolLibrary",
    "KeymakerProtocolManifest",
    "ProtocolArtifact",
    "ProtocolRoute",
    "resolve_keymaker_protocol",
    "load_keymaker_protocol_library",
    "validate_keymaker_protocol_manifest",
    "validate_keymaker_protocol_library",
    "keymaker_protocol_catalog",
]


if __name__ == "__main__":
    raise SystemExit(main())
