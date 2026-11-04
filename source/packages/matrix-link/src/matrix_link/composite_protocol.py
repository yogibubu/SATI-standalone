"""Immutable composite-gradient protocol contracts shared by LINK and KeyMaker."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


COMPOSITE_PROTOCOL_SCHEMA = "matrix.link.composite_protocol.v1"
_COMPOSITE_PROTOCOL_IDENTITIES = {
    "l2-mp2-g16": ("l2mp2", "g16"),
    "pl2-mp2-orca": ("pl2mp2", "orca"),
    "pl2-f12-orca": ("pl2f12", "orca"),
    "pl2-f12-molpro": ("pl2f12", "molpro"),
    "fpa-q5-t-cv-molpro": ("fpaq5tcv", "molpro"),
    "fpa-haq5-hat-cv-molpro": ("fpahaq5hatcv", "molpro"),
    "sherrill-tq-dz-cv-molpro": ("sherrilltqdzcv", "molpro"),
    "l2fpa-lite-molpro": ("l2fpalite", "molpro"),
    "l2fpa-molpro": ("l2fpa", "molpro"),
    "l2f12-may-lite-molpro": ("l2f12maylite", "molpro"),
    "cheap-lite-molpro": ("cheaplite", "molpro"),
    "l2-molpro": ("l2", "molpro"),
    "pl2-orca": ("pl2", "orca"),
    "fast-molpro": ("fast", "molpro"),
    "pl2-lite-f12-molpro": ("pl2litef12", "molpro"),
    "pl2-lite-mp2-molpro": ("pl2litemp2(tzqz)", "molpro"),
    "pl2-lite-db-molpro": ("pl2litedb", "molpro"),
    "l2-lite-db-molpro": ("l2litedb", "molpro"),
    "l2-lite-f12-molpro": ("l2litef12", "molpro"),
    "l2f12-lite-hybrid-molpro": ("l2f12lite", "molpro"),
    "pl2f12-lite-hybrid-molpro": ("pl2f12lite", "molpro"),
    "pl2f12-may-lite-molpro": ("pl2f12maylite", "molpro"),
    "pfpa-lite-molpro": ("pfpalite", "molpro"),
    "pcs2-orca": ("pcs2", "orca"),
    "ppcs2-orca": ("ppcs2", "orca"),
    "pcs2-molpro": ("l2f12", "molpro"),
    "l2f12-may-molpro": ("l2f12may", "molpro"),
    "ppcs2-molpro": ("ppcs2", "molpro"),
    "pcs2-mrcc": ("pcs2", "mrcc"),
    "ppcs2-mrcc": ("ppcs2", "mrcc"),
    "junchs-gdv": ("junchs", "gdv"),
    "junchs-cfour": ("junchs", "cfour"),
    "junchs-molpro": ("junchs", "molpro"),
}
# Release lock for the scientific library. Editing a protocol JSON is
# fail-closed until an explicitly requested new version is reviewed and pinned.
_PINNED_PROTOCOL_SHA256 = {
    "l2-mp2-g16": "58d4c74006437555c1f0be59c996c595ea1da3fc8249acd4a6f3e2ffcfcc744b",
    "pl2-mp2-orca": "06247a27bc994d01e1b6a784d48c19151d099bdecaf843a8cd18bc18a54bb640",
    "pl2-f12-orca": "5ef8bea791793c2e213a0b859f01577751c595012476e7cf42af50dfa1e15d65",
    "pl2-f12-molpro": "418304d37c66ce0351401de74a6f2f7f427aa88e7c5897084c6c207148ec16e8",
    "fpa-q5-t-cv-molpro": "72bbb38c6f4d016e76a8c09bd3b45058887dcc3d6f3a243c96284986c43e8923",
    "fpa-haq5-hat-cv-molpro": "06cc5a230a5ab57d9024b4a109a3575a06240368fddd0839a711e95973fbe3ad",
    "sherrill-tq-dz-cv-molpro": "cc0a7f5bddc6b494f5e5ee39db0c7f76e26d30d87e6864ad48bad93c676789b2",
    "l2fpa-lite-molpro": "a8e0c111f98d4b61ba3c2b22c1b29bf902ef8a6e606b12bf51b17c7133db8b3c",
    "l2fpa-molpro": "627a7b903ed6b1b4cbc2bbf67943b30393d1decf1f8dc88d1313e0de4c0e1439",
    "l2f12-may-lite-molpro": "b7067d1ba3584e9503edb68ddda0d8455de21f96a5ab020fbcd813e0354182ea",
    "cheap-lite-molpro": "663cc812f8315a91fcb4105a06243e97af555ef6bc6b3475e7250741eb3d80fe",
    "l2-molpro": "7735f9698fac318da7637505b528f8e629e3a35a770282b97f78dc6426eaf5e7",
    "pl2-orca": "2bdb989a18aa6fe707715761b55ccb699304002cc65309dcc7084eb58ac10e9a",
    "fast-molpro": "3c4c9f91d595158e0eb0e33f508a537162e1dde74369e684645deeeaac4bb4d4",
    "pl2-lite-f12-molpro": "2f14e9f7dc4634e1ca3220db238d84c13ee36529d3d3bb9521f1363ec69f2ff1",
    "pl2-lite-mp2-molpro": "ee1221c65c70af8b86d4077d39ce2260be56a28bfdc50845bd95fc2bbfab6f4f",
    "pl2-lite-db-molpro": "86e9cb1bf50231a926eb7fc2dcdbf5a3a98f44ac72552631156b00ce824b2e66",
    "l2-lite-db-molpro": "532855aa9094e2fc48a25ee483f4189e5d33fd0d61f10f0cb710910352d425a9",
    "l2-lite-f12-molpro": "9915842d41f858cc20fa39af39f4060b00559bfe7b6bbb708b15c5d042e96235",
    "l2f12-lite-hybrid-molpro": "d865d2d234d148c67576fecb9c16a8fdb08bc07e096f5954f9abad6359218502",
    "pl2f12-lite-hybrid-molpro": "26b6087c66e040a16a119d68ac643d5674bb01f070c817e89d27c7217eae9f0f",
    "pl2f12-may-lite-molpro": "c6fef139019c935fcc72cda18a4ef830b763b386a6409fb6b177a101cb5ac755",
    "pfpa-lite-molpro": "a2c19714e21da8471f5bb0cd1329c9c94927cf946ace4b043e6d2ff1b67b02fb",
    "junchs-gdv": "d8cb3e4c6c9d6cf8b8418576bbd02eb3e207f2fcd92c4ca6fe091206c026788c",
    "junchs-cfour": "2c62e24dd12033190047f5f29971310d5f26b01dbb47757ffa1bbfc1f9f75ea5",
    "junchs-molpro": "ee0f22440f14bdcf53ad8ad06c97b31304b30d20fbde3d4dcceae0157504104f",
    "pcs2-molpro": "7312115fcb95623f982815e9a5f5f16a46f6a2fc9602702e5b97672ca3c0c7dd",
    "l2f12-may-molpro": "cc4edb3e9e84c11da9c23a6c75cc6e9de7b1526a20367b2488675c697e9bd3f4",
    "pcs2-mrcc": "7f6c9cef7e0dec9554d5f369f9f88fa2381221db4353d3e815558079d60ae2b0",
    "pcs2-orca": "586c1e1f41f5c7f49bac3a95a5f23905eba0977ee81ede4873cf4eb6ea0e59b1",
    "ppcs2-molpro": "e4cd89db3ee71a29a2dfcc9664ccdc713a9e0b2931eda37ac4e42839bac24cb3",
    "ppcs2-mrcc": "c3a1d34538ab0b21ec0f43137347efaa1a3352432b0cd5c26259f090d6fc1583",
    "ppcs2-orca": "471bee2b1fb2691f4530742f21b73d0f78e108bd7853a57797e484c426540c54",
}
COMPOSITE_PROTOCOL_SELECTORS = tuple(_COMPOSITE_PROTOCOL_IDENTITIES)
_FPA_MOLPRO_SELECTORS = {
    "fpa-q5-t-cv-molpro",
    "fpa-haq5-hat-cv-molpro",
    "sherrill-tq-dz-cv-molpro",
    "l2fpa-lite-molpro",
    "l2fpa-molpro",
    "cheap-lite-molpro",
}
_PL2_F12_SELECTORS = {"pl2-f12-orca", "pl2-f12-molpro"}
_LITE_MOLPRO_SELECTORS = {
    "pl2-lite-f12-molpro",
    "pl2-lite-mp2-molpro",
    "pl2-lite-db-molpro",
    "l2-lite-db-molpro",
    "l2-lite-f12-molpro",
}
_LITE_F12_MOLPRO_SELECTORS = {
    "pl2-lite-f12-molpro",
    "l2-lite-f12-molpro",
}
_FINAL_LITE_MOLPRO_SELECTORS = {
    "l2f12-lite-hybrid-molpro",
    "pl2f12-lite-hybrid-molpro",
    "pfpa-lite-molpro",
    "l2f12-may-lite-molpro",
    "pl2f12-may-lite-molpro",
}
_EMPIRICAL_CV_SELECTORS = {
    "fast-molpro",
    "l2fpa-lite-molpro",
    "l2f12-may-lite-molpro",
    "cheap-lite-molpro",
    *_LITE_MOLPRO_SELECTORS,
    *_FINAL_LITE_MOLPRO_SELECTORS,
}
_ANALYTIC_HIGH_LEVEL_SELECTORS = {
    "junchs-cfour",
    "junchs-molpro",
    "l2-molpro",
    "fast-molpro",
    *_FPA_MOLPRO_SELECTORS,
}

_DATA = Path(__file__).resolve().parent / "data"
_PROTOCOL_DATA = _DATA / "composite_protocols"
_BASIS_DATA = _DATA / "basis"


@dataclass(frozen=True)
class CompositeProtocol:
    """Validated, content-addressed composite protocol."""

    selector: str
    protocol_id: str
    model: str
    backend: str
    payload: dict[str, Any]
    path: Path
    sha256: str
    basis_path: Path
    core_valence_basis_path: Path
    cbs_lower_basis_path: Path
    cbs_upper_basis_path: Path


@dataclass(frozen=True)
class CompositeGradientContract:
    """Optimizer-facing derivative settings fixed by a composite protocol."""

    mode: str
    numerical_step_bohr: float
    numerical_stencil: str
    uses_numerical_gradient: bool


def _normalize_selector(value: str) -> str:
    return str(value).strip().casefold().replace("_", "-")


def _normalize_model(value: str) -> str:
    key = (
        _normalize_selector(value)
        .replace("-", "")
        .replace("/", "")
        .replace("+", "")
    )
    return {"juncheap": "junchs"}.get(key, key)


def _required_numerical_step_bohr(high_level: dict[str, Any]) -> float:
    """Return the method-class finite-difference step fixed by the library."""

    correlation_model = str(high_level.get("correlation_model", "")).strip().upper()
    method = str(high_level.get("method", "")).strip().upper()
    if correlation_model in {"PNO", "DLPNO"} and "F12" in method:
        return 3.0e-3
    return 1.0e-3


def resolve_composite_protocol_selector(model: str, backend: str) -> str:
    """Resolve one exact model/backend pair and reject ambiguous requests."""

    selector_request = _normalize_selector(model)
    model_key = _normalize_model(model)
    backend_key = _normalize_selector(backend)
    final_taxonomy = {
        "fl2": {"": "fast-molpro", "auto": "fast-molpro", "molpro": "fast-molpro"},
        "l2mp2": {
            "": "l2-mp2-g16",
            "auto": "l2-mp2-g16",
            "g16": "l2-mp2-g16",
            "gaussian": "l2-mp2-g16",
            "cfour": "junchs-cfour",
            "molpro": "junchs-molpro",
        },
        "l2f12": {
            "": "pcs2-orca",
            "auto": "pcs2-orca",
            "orca": "pcs2-orca",
            "molpro": "pcs2-molpro",
        },
        "pl2mp2": {
            "": "pl2-mp2-orca",
            "auto": "pl2-mp2-orca",
            "orca": "pl2-mp2-orca",
        },
        "pl2f12": {
            "": "pl2-f12-orca",
            "auto": "pl2-f12-orca",
            "orca": "pl2-f12-orca",
            "molpro": "pl2-f12-molpro",
        },
    }
    # An exact frozen selector supplied as the backend is less ambiguous than
    # a family/backend request and must be validated before family routing.
    # This is the form emitted by KeyMaker after the menu choice has already
    # been resolved to one immutable protocol.
    if backend_key in COMPOSITE_PROTOCOL_SELECTORS:
        expected_model, _expected_backend = _COMPOSITE_PROTOCOL_IDENTITIES[backend_key]
        if model_key not in {"", expected_model}:
            raise ValueError(
                f"backend {backend_key!r} requires model {expected_model}"
            )
        return backend_key
    if model_key in final_taxonomy:
        try:
            return final_taxonomy[model_key][backend_key]
        except KeyError as exc:
            raise ValueError(
                f"{model!s} has no frozen protocol for backend {backend!r}"
            ) from exc
    if selector_request in COMPOSITE_PROTOCOL_SELECTORS:
        expected_model, expected_backend = _COMPOSITE_PROTOCOL_IDENTITIES[selector_request]
        if backend_key not in {"", "auto", expected_backend, selector_request}:
            raise ValueError(
                f"composite selector {selector_request!r} conflicts with backend {backend!r}"
            )
        return selector_request
    supported_models = {identity[0] for identity in _COMPOSITE_PROTOCOL_IDENTITIES.values()}
    if model_key not in supported_models:
        raise ValueError(f"unsupported composite model: {model!r}")
    if backend_key in {"", "auto"}:
        if model_key == "junchs":
            return "junchs-gdv"
        model_matches = tuple(
            selector
            for selector, identity in _COMPOSITE_PROTOCOL_IDENTITIES.items()
            if identity[0] == model_key
        )
        if len(model_matches) == 1:
            return model_matches[0]
        raise ValueError(
            f"{model!s} is backend-specific; choose an explicit supported backend"
        )
    matches = tuple(
        selector
        for selector, identity in _COMPOSITE_PROTOCOL_IDENTITIES.items()
        if identity == (model_key, backend_key)
    )
    if len(matches) != 1:
        raise ValueError(
            f"{model!s} has no frozen protocol for backend {backend!r}"
        )
    return matches[0]


def load_composite_protocol(selector: str) -> CompositeProtocol:
    """Load and validate one packaged protocol, including its basis artifact."""

    key = _normalize_selector(selector)
    if key not in COMPOSITE_PROTOCOL_SELECTORS:
        raise ValueError(f"unsupported composite protocol selector: {selector!r}")
    path = _PROTOCOL_DATA / f"{key}.json"
    raw = path.read_bytes()
    protocol_sha256 = hashlib.sha256(raw).hexdigest()
    if protocol_sha256 != _PINNED_PROTOCOL_SHA256[key]:
        raise RuntimeError(
            f"frozen composite protocol {key} differs from the pinned library release"
        )
    payload = json.loads(raw)
    expected_model, expected_backend = _COMPOSITE_PROTOCOL_IDENTITIES[key]
    required = {
        "schema",
        "protocol_id",
        "selector",
        "model",
        "backend",
        "frozen",
        "formula",
        "orbital_basis",
        "high_level",
        "initialization",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"composite protocol {key} lacks: {', '.join(missing)}")
    if payload["schema"] != COMPOSITE_PROTOCOL_SCHEMA:
        raise RuntimeError(f"composite protocol {key} has an unsupported schema")
    if payload["selector"] != key:
        raise RuntimeError(f"composite protocol selector mismatch in {path}")
    if _normalize_model(payload["model"]) != expected_model:
        raise RuntimeError(f"composite protocol model mismatch in {path}")
    if _normalize_selector(payload["backend"]) != expected_backend:
        raise RuntimeError(f"composite protocol backend mismatch in {path}")
    if payload["frozen"] is not True:
        raise RuntimeError(f"composite protocol {key} is not frozen")
    high = payload["high_level"]
    initialization = payload["initialization"]
    derivative_contract = "one_sided_totally_symmetric_sonic_numerical"
    required_numerical_step = _required_numerical_step_bohr(high)
    if key in _ANALYTIC_HIGH_LEVEL_SELECTORS:
        if high.get("derivative") != "analytic":
            raise RuntimeError(f"composite protocol {key} must use an analytic CCSD(T) gradient")
    else:
        if high.get("derivative") != derivative_contract:
            raise RuntimeError(
                f"composite protocol {key} must use one-sided totally symmetric SONIC derivatives"
            )
        if (
            float(high.get("numerical_gradient_step_bohr", 0.0))
            != required_numerical_step
        ):
            raise RuntimeError(
                f"composite protocol {key} has an invalid method-class numerical gradient step"
            )
    if key in {"junchs-gdv", "junchs-cfour", "junchs-molpro", "l2-mp2-g16"}:
        cbs = payload.get("cbs_correction", {})
        lower = cbs.get("lower", {})
        upper = cbs.get("upper", {})
        if (
            int(cbs.get("exponent", 0)) != 3
            or int(cbs.get("lower_cardinal", 0)) != 3
            or int(cbs.get("upper_cardinal", 0)) != 4
        ):
            raise RuntimeError(f"composite protocol {key} changed the MP2 CBS extrapolation")
        expected_coefficient = 64.0 / 37.0
        if abs(float(lower.get("coefficient", 0.0)) + expected_coefficient) > 1.0e-15:
            raise RuntimeError(f"composite protocol {key} changed the jun-TZ coefficient")
        if abs(float(upper.get("coefficient", 0.0)) - expected_coefficient) > 1.0e-15:
            raise RuntimeError(f"composite protocol {key} changed the jun-QZ coefficient")
        if lower.get("basis") != "jun-cc-pVTZ" or upper.get("basis") != "jun-cc-pVQZ":
            raise RuntimeError(f"composite protocol {key} changed the MP2 CBS bases")
        if lower.get("derivative") != "analytic" or upper.get("derivative") != "analytic":
            raise RuntimeError(f"composite protocol {key} must use analytic MP2 CBS gradients")
    elif key == "pl2-mp2-orca":
        local = payload.get("local_mp2_correction", {})
        conventional = local.get("conventional", {})
        local_mp2 = local.get("local", {})
        cbs = payload.get("cbs_correction", {})
        lower = cbs.get("lower", {})
        upper = cbs.get("upper", {})
        expected = 64.0 / 37.0
        if (
            conventional.get("basis") != "def2-TZVPP"
            or local_mp2.get("basis") != "def2-TZVPP"
            or conventional.get("derivative") != "analytic"
            or local_mp2.get("derivative") != "analytic"
            or int(cbs.get("exponent", 0)) != 3
            or int(cbs.get("lower_cardinal", 0)) != 3
            or int(cbs.get("upper_cardinal", 0)) != 4
            or lower.get("basis") != "def2-TZVPP"
            or upper.get("basis") != "def2-QZVPP"
            or abs(float(lower.get("coefficient", 0.0)) + expected) > 1.0e-15
            or abs(float(upper.get("coefficient", 0.0)) - expected) > 1.0e-15
            or lower.get("derivative") != "analytic"
            or upper.get("derivative") != "analytic"
        ):
            raise RuntimeError(f"composite protocol {key} changed its local-MP2/CBS contract")
    elif key in _FPA_MOLPRO_SELECTORS:
        focal_point = payload.get("focal_point", {})
        terms = focal_point.get("terms", [])
        total_mp2_cbs = key in {"l2fpa-lite-molpro", "l2fpa-molpro", "cheap-lite-molpro"}
        expected_roles = (
            ["high_level", "cbs_mp2_lower", "cbs_mp2_upper"]
            if key == "cheap-lite-molpro"
            else ["high_level", "delta_mp2", "cbs_mp2_lower", "cbs_mp2_upper"]
            if key in {"l2fpa-lite-molpro", "l2fpa-molpro"}
            else [
                "high_level",
                "delta_mp2",
                "cbs_mp2_lower",
                "cbs_hf_lower",
                "cbs_mp2_upper",
                "cbs_hf_upper",
            ]
        )
        lower_cardinal, upper_cardinal = (
            (3, 4)
            if key in {"sherrill-tq-dz-cv-molpro", "l2fpa-lite-molpro", "l2fpa-molpro", "cheap-lite-molpro"}
            else (4, 5)
        )
        if (
            int(focal_point.get("exponent", 0)) != 3
            or int(focal_point.get("lower_cardinal", 0)) != lower_cardinal
            or int(focal_point.get("upper_cardinal", 0)) != upper_cardinal
            or focal_point.get("scf_source")
            != (
                "coextrapolated_with_total_mp2_energy"
                if total_mp2_cbs
                else "upper_cardinal_basis"
            )
            or [term.get("role") for term in terms] != expected_roles
        ):
            raise RuntimeError(f"composite protocol {key} changed the Q5/T focal-point contract")
        expected_coefficients = (
            [1.0, -64.0 / 37.0, 64.0 / 37.0]
            if key == "cheap-lite-molpro"
            else [1.0, -1.0, -27.0 / 37.0, 64.0 / 37.0]
            if key in {"l2fpa-lite-molpro", "l2fpa-molpro"}
            else [1.0, -1.0, -27.0 / 37.0, 27.0 / 37.0, 64.0 / 37.0, -27.0 / 37.0]
            if key == "sherrill-tq-dz-cv-molpro"
            else [1.0, -1.0, -64.0 / 61.0, 64.0 / 61.0, 125.0 / 61.0, -64.0 / 61.0]
        )
        if any(
            abs(float(term.get("coefficient", 0.0)) - expected) > 1.0e-15
            for term, expected in zip(terms, expected_coefficients, strict=True)
        ):
            raise RuntimeError(f"composite protocol {key} changed its focal-point coefficients")
        if any(term.get("derivative") != "analytic" for term in terms):
            raise RuntimeError(f"composite protocol {key} requires analytic term gradients")
        for term in terms:
            artifact = str(term.get("artifact", ""))
            artifact_path = _BASIS_DATA / artifact if artifact else Path()
            if not artifact or not artifact_path.is_file():
                raise RuntimeError(
                    f"composite protocol {key} lacks basis artifact for {term.get('role')}"
                )
            if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != term.get("sha256"):
                raise RuntimeError(
                    f"composite protocol {key} basis hash mismatch for {term.get('role')}"
                )
    elif key == "ppcs2-mrcc":
        correction = payload.get("f12_correction", {})
        cv = payload.get("core_valence", {})
        if correction.get("explicitly_correlated", {}).get("derivative") != derivative_contract:
            raise RuntimeError(f"composite protocol {key} must numerically differentiate MP2-F12")
        if correction.get("conventional", {}).get("derivative") != "analytic":
            raise RuntimeError(f"composite protocol {key} must use an analytic MP2 gradient")
        if cv.get("derivative") != "analytic":
            raise RuntimeError(f"composite protocol {key} must use analytic core-valence MP2 gradients")
    elif key in _PL2_F12_SELECTORS:
        correction = payload.get("local_mp2_f12_correction", {})
        conventional = correction.get("conventional", {})
        local = correction.get("local", {})
        expected_methods = (
            {"conventional": "F12-RI-MP2", "local": "DLPNO-MP2-F12/D"}
            if key.endswith("orca")
            else {"conventional": "MP2-F12", "local": "PNO-LMP2-F12"}
        )
        if (
            payload.get("cbs_correction") is not None
            or conventional.get("basis") != "may-cc-pVTZ"
            or local.get("basis") != "may-cc-pVTZ"
            or conventional.get("method") != expected_methods["conventional"]
            or local.get("method") != expected_methods["local"]
            or conventional.get("derivative") != derivative_contract
            or local.get("derivative") != derivative_contract
            or float(conventional.get("numerical_gradient_step_bohr", 0.0))
            != required_numerical_step
            or float(local.get("numerical_gradient_step_bohr", 0.0))
            != required_numerical_step
        ):
            raise RuntimeError(
                f"composite protocol {key} changed its no-CBS conventional-minus-local MP2-F12 recovery"
            )
    if key == "fast-molpro":
        density_based = payload.get("density_based_correction", {})
        if density_based.get("method") != "DF-BASCORR(PBEC)":
            raise RuntimeError(f"composite protocol {key} changed its density-based method")
        if density_based.get("functional") != "PBEC":
            raise RuntimeError(f"composite protocol {key} changed its BASCORR functional")
        if density_based.get("derivative") != derivative_contract:
            raise RuntimeError(
                f"composite protocol {key} must numerically differentiate the BASCORR term"
            )
        if float(density_based.get("numerical_gradient_step_bohr", 0.0)) != 1.0e-3:
            raise RuntimeError(f"composite protocol {key} has an invalid BASCORR step")
    if key in _LITE_MOLPRO_SELECTORS:
        if high.get("basis") != "cc-pV(T+d)Z":
            raise RuntimeError(f"composite protocol {key} changed its triple-zeta basis")
        if key in _LITE_F12_MOLPRO_SELECTORS:
            expected_method = (
                "PNO-LCCSD(T)-F12,SCALE_TRIP=1,CABS_SINGLES=1"
                if key.startswith("pl2-")
                else "CCSD(T)-F12b,SCALE_TRIP=1,CABS_SINGLES=1"
            )
            if (
                high.get("method") != expected_method
                or high.get("f12_ansatz") != "F12b"
                or high.get("triples") != "T*"
                or high.get("cabs_singles") != "included_in_molpro_total_energy"
                or payload.get("hf_reference") is not None
                or payload.get("density_based_correction") is not None
            ):
                raise RuntimeError(f"composite protocol {key} changed its F12+CABS contract")
        elif key == "pl2-lite-mp2-molpro":
            mp2_reference = payload.get("mp2_reference", {})
            if (
                high.get("method") != "DELTA-PNO-LCCSD(T)-MP2"
                or high.get("triples") != "(T)"
                or mp2_reference.get("method") != "MP2"
                or mp2_reference.get("basis") != "cc-pV(Q+d)Z"
                or mp2_reference.get("derivative") != "analytic"
                or float(mp2_reference.get("coefficient", 0.0)) != 1.0
                or payload.get("hf_reference") is not None
                or payload.get("density_based_correction") is not None
            ):
                raise RuntimeError(f"composite protocol {key} changed its PNO/TZ plus MP2/QZ contract")
            mp2_artifact = _BASIS_DATA / str(mp2_reference.get("artifact", ""))
            if not mp2_artifact.is_file():
                raise RuntimeError(f"composite protocol {key} lacks its MP2/QZ basis artifact")
            if hashlib.sha256(mp2_artifact.read_bytes()).hexdigest() != mp2_reference.get("sha256"):
                raise RuntimeError(f"composite protocol {key} MP2/QZ basis hash mismatch")
        else:
            hf_reference = payload.get("hf_reference", {})
            expected_high_method = (
                "DELTA-CCSD(T)-HF"
                if key == "l2-lite-db-molpro"
                else "DELTA-PNO-LCCSD(T)-HF,DOMOPT=TIGHT"
                if key == "pl2-lite-db-molpro"
                else "DELTA-PNO-LCCSD(T)-HF"
            )
            if (
                high.get("method") != expected_high_method
                or high.get("triples") != "(T)"
                or (
                    key == "pl2-lite-db-molpro"
                    and high.get("domain_optimization") != "TIGHT"
                )
                or hf_reference.get("method") != "HF"
                or hf_reference.get("basis") != "cc-pV(Q+d)Z"
                or hf_reference.get("derivative") != "analytic"
                or float(hf_reference.get("coefficient", 0.0)) != 1.0
            ):
                raise RuntimeError(f"composite protocol {key} changed its CC/TZ plus HF/QZ contract")
            hf_artifact = _BASIS_DATA / str(hf_reference.get("artifact", ""))
            if not hf_artifact.is_file():
                raise RuntimeError(f"composite protocol {key} lacks its HF/QZ basis artifact")
            if hashlib.sha256(hf_artifact.read_bytes()).hexdigest() != hf_reference.get("sha256"):
                raise RuntimeError(f"composite protocol {key} HF/QZ basis hash mismatch")
            density_based = payload.get("density_based_correction")
            if key in {"pl2-lite-db-molpro", "l2-lite-db-molpro"}:
                if (
                    not isinstance(density_based, dict)
                    or density_based.get("method") != "DF-BASCORR(PBEC)"
                    or density_based.get("basis") != "cc-pV(T+d)Z"
                    or density_based.get("functional") != "PBEC"
                    or density_based.get("derivative") != derivative_contract
                    or float(density_based.get("numerical_gradient_step_bohr", 0.0)) != 1.0e-3
                ):
                    raise RuntimeError(f"composite protocol {key} changed its density-based correction")
            elif density_based is not None:
                raise RuntimeError(f"composite protocol {key} unexpectedly contains a density correction")
    cv = payload.get("core_valence", {})
    if cv.get("derivative") != "analytic":
        raise RuntimeError(f"composite protocol {key} must use analytic MP2 gradients")
    if initialization.get("geometry") != "B3LYP/6-31G*" or initialization.get(
        "hessian"
    ) != "B3LYP/6-31G*":
        raise RuntimeError(f"composite protocol {key} has an invalid initialization level")
    if key in {
        "l2-mp2-g16",
        "junchs-gdv",
        "junchs-cfour",
        "junchs-molpro",
        *_FPA_MOLPRO_SELECTORS,
    } and initialization.get("backend") not in {"gdv", "g16"}:
        raise RuntimeError(f"composite protocol {key} must initialize with G16.C02")
    basis = payload["orbital_basis"]
    if key == "fpa-q5-t-cv-molpro":
        expected_basis = "cc-pV(T+d)Z"
    elif key == "fpa-haq5-hat-cv-molpro":
        expected_basis = "heavy-aug-cc-pV(T+d)Z"
    elif key in {"sherrill-tq-dz-cv-molpro", "l2fpa-lite-molpro", "l2fpa-molpro"}:
        expected_basis = "cc-pV(D+d)Z"
    elif key == "cheap-lite-molpro":
        expected_basis = "cc-pV(T+d)Z"
    elif key in {"l2-molpro", "pl2-orca", "pl2-mp2-orca"}:
        expected_basis = "def2-TZVPP"
    elif key == "fast-molpro":
        expected_basis = "def2-SVP"
    elif key in {"l2f12-lite-hybrid-molpro", "pl2f12-lite-hybrid-molpro"}:
        expected_basis = "cc-pVDZ-F12_may-cc-pVTZ"
    elif key in {"l2f12-may-lite-molpro", "pl2f12-may-lite-molpro"}:
        expected_basis = "may-cc-pVTZ"
    elif key == "pfpa-lite-molpro":
        expected_basis = "cc-pV(D+d)Z"
    elif key in _LITE_MOLPRO_SELECTORS or key == "pcs2-molpro":
        expected_basis = "cc-pV(T+d)Z"
    elif key == "l2f12-may-molpro":
        expected_basis = "may-cc-pVTZ"
    else:
        expected_basis = (
            "jun-cc-pVTZ"
            if key in {
                "ppcs2-mrcc",
                "junchs-gdv",
                "junchs-cfour",
                "junchs-molpro",
                "l2-mp2-g16",
            }
            else "may-cc-pVTZ"
        )
    if basis.get("name") != expected_basis:
        raise RuntimeError(f"composite protocol {key} has an invalid orbital basis")
    if high.get("basis") != expected_basis:
        raise RuntimeError(f"composite protocol {key} high-level term changed orbital basis")
    cv = payload.get("core_valence", {})
    if key in _EMPIRICAL_CV_SELECTORS:
        if cv.get("type") != "oracle_cv_exponential_field":
            raise RuntimeError(f"composite protocol {key} changed its empirical CV field")
    elif cv.get("basis") != "cc-pwCVTZ" or cv.get("hydrogen_basis") != "cc-pVTZ":
        raise RuntimeError(
            f"composite protocol {key} must use cc-pwCVTZ core-valence basis "
            "and cc-pVTZ on H/He"
        )
    if key == "ppcs2-mrcc":
        correction = payload.get("f12_correction", {})
        if correction.get("explicitly_correlated", {}).get("basis") != expected_basis:
            raise RuntimeError(f"composite protocol {key} MP2-F12 basis changed")
        if correction.get("conventional", {}).get("basis") != expected_basis:
            raise RuntimeError(f"composite protocol {key} conventional MP2 basis changed")
    artifact = str(basis.get("artifact", ""))
    basis_path = _BASIS_DATA / artifact if artifact else Path()
    if artifact:
        if not basis_path.is_file():
            raise RuntimeError(f"composite protocol basis artifact is missing: {basis_path}")
        basis_sha256 = hashlib.sha256(basis_path.read_bytes()).hexdigest()
        if basis_sha256 != basis.get("sha256"):
            raise RuntimeError(f"composite protocol basis hash mismatch: {basis_path}")
    elif (expected_backend, basis.get("source")) not in {
        ("mrcc", "MRCC native basis library"),
        ("gdv", "GDV native basis library"),
    }:
        raise RuntimeError(f"composite protocol {key} has no validated basis source")
    cbs_lower_basis_path = Path()
    cbs_upper_basis_path = Path()
    if key in {"junchs-molpro", "l2-mp2-g16", "pl2-mp2-orca"}:
        for label, spec in (("lower", payload["cbs_correction"]["lower"]), ("upper", payload["cbs_correction"]["upper"])):
            cbs_artifact = str(spec.get("artifact", ""))
            cbs_path = _BASIS_DATA / cbs_artifact if cbs_artifact else Path()
            if not cbs_artifact or not cbs_path.is_file():
                raise RuntimeError(f"composite protocol {key} lacks its {label} CBS basis artifact")
            if hashlib.sha256(cbs_path.read_bytes()).hexdigest() != spec.get("sha256"):
                raise RuntimeError(f"composite protocol {key} {label} CBS basis hash mismatch")
            if label == "lower":
                cbs_lower_basis_path = cbs_path
            else:
                cbs_upper_basis_path = cbs_path
    cv_artifact = str(cv.get("artifact", ""))
    cv_basis_path = _BASIS_DATA / cv_artifact if cv_artifact else Path()
    if cv_artifact:
        if not cv_basis_path.is_file():
            raise RuntimeError(
                f"composite protocol core-valence basis artifact is missing: {cv_basis_path}"
            )
        cv_basis_sha256 = hashlib.sha256(cv_basis_path.read_bytes()).hexdigest()
        if cv_basis_sha256 != cv.get("sha256"):
            raise RuntimeError(
                f"composite protocol core-valence basis hash mismatch: {cv_basis_path}"
            )
    if key in {"junchs-gdv", "junchs-cfour", "l2-mp2-g16"} and not cv_artifact:
        raise RuntimeError(f"composite protocol {key} requires an explicit core-valence basis")
    return CompositeProtocol(
        selector=key,
        protocol_id=str(payload["protocol_id"]),
        model=str(payload["model"]),
        backend=str(payload["backend"]),
        payload=payload,
        path=path,
        sha256=protocol_sha256,
        basis_path=basis_path,
        core_valence_basis_path=cv_basis_path,
        cbs_lower_basis_path=cbs_lower_basis_path,
        cbs_upper_basis_path=cbs_upper_basis_path,
    )


def composite_protocol_gradient_contract(
    protocol: CompositeProtocol | str,
) -> CompositeGradientContract:
    """Return immutable optimizer settings implied by the term derivatives."""

    resolved = load_composite_protocol(protocol) if isinstance(protocol, str) else protocol
    high = resolved.payload["high_level"]
    numerical_terms = (
        high["derivative"] != "analytic"
        or resolved.payload.get("density_based_correction", {}).get("derivative")
        == "one_sided_totally_symmetric_sonic_numerical"
    )
    if not numerical_terms:
        return CompositeGradientContract(
            mode="analytic",
            numerical_step_bohr=1.0e-3,
            numerical_stencil="forward",
            uses_numerical_gradient=False,
        )
    return CompositeGradientContract(
        mode="hybrid",
        numerical_step_bohr=float(
            high["numerical_gradient_step_bohr"]
            if "numerical_gradient_step_bohr" in high
            else resolved.payload["density_based_correction"][
                "numerical_gradient_step_bohr"
            ]
        ),
        numerical_stencil="forward",
        uses_numerical_gradient=True,
    )


def composite_protocol_backend(selector: str, **kwargs: Any):
    """Construct a backend exclusively from its validated frozen protocol."""

    protocol = load_composite_protocol(selector)
    from .ppcs2 import (
        junchs_cfour_backend,
        junchs_gdv_backend,
        junchs_molpro_backend,
        fast_molpro_backend,
        lite_molpro_backend,
        l2_molpro_backend,
        pl2_orca_backend,
        pl2_mp2_orca_backend,
        pcs2_molpro_backend,
        pcs_orca_backend,
        ppcs2_molpro_backend,
    )

    if protocol.selector == "l2-mp2-g16":
        return junchs_gdv_backend(protocol_selector=protocol.selector, **kwargs)
    if protocol.selector == "pl2-mp2-orca":
        return pl2_mp2_orca_backend(**kwargs)

    if protocol.selector == "junchs-gdv":
        return junchs_gdv_backend(**kwargs)
    if protocol.selector == "junchs-cfour":
        return junchs_cfour_backend(**kwargs)
    if protocol.selector == "junchs-molpro":
        return junchs_molpro_backend(**kwargs)
    if protocol.selector == "l2-molpro":
        return l2_molpro_backend(**kwargs)
    if protocol.selector == "pl2-orca":
        return pl2_orca_backend(**kwargs)
    if protocol.selector == "fast-molpro":
        return fast_molpro_backend(**kwargs)
    if protocol.selector in _LITE_MOLPRO_SELECTORS | _FINAL_LITE_MOLPRO_SELECTORS:
        return lite_molpro_backend(protocol_selector=protocol.selector, **kwargs)
    if protocol.selector in _FPA_MOLPRO_SELECTORS:
        from .ppcs2 import fpa_molpro_backend

        return fpa_molpro_backend(protocol.selector, **kwargs)

    if protocol.selector == "pcs2-orca":
        return pcs_orca_backend(coupled_cluster_approximation="canonical", **kwargs)
    if protocol.selector in {"ppcs2-orca", "pl2-f12-orca"}:
        return pcs_orca_backend(
            coupled_cluster_approximation="dlpno",
            protocol_selector=protocol.selector,
            **kwargs,
        )
    if protocol.selector in {"pcs2-molpro", "l2f12-may-molpro"}:
        return pcs2_molpro_backend(protocol_selector=protocol.selector, **kwargs)
    if protocol.selector in {"ppcs2-molpro", "pl2-f12-molpro"}:
        return ppcs2_molpro_backend(protocol_selector=protocol.selector, **kwargs)
    from .ppcs2 import pcs_mrcc_backend, ppcs2_mrcc_backend

    if protocol.selector == "pcs2-mrcc":
        return pcs_mrcc_backend(**kwargs)
    return ppcs2_mrcc_backend(**kwargs)


__all__ = [
    "COMPOSITE_PROTOCOL_SCHEMA",
    "COMPOSITE_PROTOCOL_SELECTORS",
    "CompositeProtocol",
    "CompositeGradientContract",
    "composite_protocol_backend",
    "composite_protocol_gradient_contract",
    "load_composite_protocol",
    "resolve_composite_protocol_selector",
]
