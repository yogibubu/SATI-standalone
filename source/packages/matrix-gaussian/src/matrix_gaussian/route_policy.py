"""Immutable Gaussian route policy and explicit route-bound overrides."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import re


GAUSSIAN_ROUTE_POLICY_SCHEMA = "matrix.gaussian.route-policy.v2"
GAUSSIAN_ROUTE_POLICY_VERSION = "2.0.0"
GAUSSIAN_ROUTE_POLICY_RULES = frozenset(("tight",))
_CANONICAL_POLICY = {
    "schema": GAUSSIAN_ROUTE_POLICY_SCHEMA,
    "version": GAUSSIAN_ROUTE_POLICY_VERSION,
    "default_action": "reject",
    "forbidden_rules": {
        "nosymm": {
            "description": "Disabling Gaussian symmetry is always forbidden.",
            "aliases": ["NoSymm", "NoSymmetry", "Symm=None", "Symmetry=None"],
        },
        "tight": {
            "description": "Tight Gaussian convergence settings are forbidden by default.",
            "aliases": ["Tight", "VeryTight", "TightSCF", "VeryTightSCF"],
        },
    },
    "override_policy": {
        "explicit": True,
        "route_digest_required": True,
        "reason_required": True,
        "per_rule_authorization": True,
        "non_overridable_rules": ["nosymm"],
    },
    "change_policy": "new_manifest_version_and_explicit_approval",
}
_NOSYMM_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9_])nosymm(?:etry)?(?![A-Za-z0-9_])", re.IGNORECASE),
    re.compile(
        r"\b(?:symm|symmetry)\s*=\s*(?:none|off|no)\b",
        re.IGNORECASE,
    ),
)
_TIGHT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:very)?tight(?:scf)?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


class GaussianRoutePolicyError(ValueError):
    """Raised when a Gaussian route violates the frozen MATRIX policy."""


@dataclass(frozen=True)
class GaussianRouteOverride:
    """Explicit authorization for forbidden rules in one exact route."""

    route_sha256: str
    allowed_rules: frozenset[str]
    reason: str
    authorized_by: str
    schema: str = "matrix.gaussian.route-override.v1"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", self.route_sha256):
            raise ValueError("Gaussian route override requires a SHA-256 route digest")
        unknown = set(self.allowed_rules) - set(GAUSSIAN_ROUTE_POLICY_RULES)
        if unknown:
            raise ValueError(f"unknown Gaussian route override rules: {sorted(unknown)}")
        if not self.allowed_rules:
            raise ValueError("Gaussian route override must authorize at least one rule")
        if not self.reason.strip():
            raise ValueError("Gaussian route override requires a non-empty reason")
        if not self.authorized_by.strip():
            raise ValueError("Gaussian route override requires an authorizer")


def gaussian_route_policy_manifest() -> dict[str, object]:
    """Return the validated frozen Gaussian route-policy manifest."""

    return json.loads(json.dumps(_loaded_policy(), sort_keys=True))


@lru_cache(maxsize=1)
def _loaded_policy() -> dict[str, object]:
    path = files("matrix_gaussian").joinpath("data/gaussian_route_policy_v2.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload != _CANONICAL_POLICY:
        raise RuntimeError(
            "Gaussian route policy manifest differs from the immutable code contract"
        )
    return payload


def gaussian_route_digest(route: str) -> str:
    canonical = " ".join(str(route).strip().split()).casefold()
    if not canonical:
        raise GaussianRoutePolicyError("Gaussian route cannot be empty")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def gaussian_route_violations(route: str) -> frozenset[str]:
    """Return forbidden policy rules present in a Gaussian route."""

    _loaded_policy()
    text = str(route).strip()
    if not text:
        raise GaussianRoutePolicyError("Gaussian route cannot be empty")
    rules: set[str] = set()
    if any(pattern.search(text) for pattern in _NOSYMM_PATTERNS):
        rules.add("nosymm")
    if _TIGHT_PATTERN.search(text):
        rules.add("tight")
    return frozenset(rules)


def create_gaussian_route_override(
    route: str,
    *,
    allow_nosymm: bool = False,
    allow_tight: bool = False,
    reason: str,
    authorized_by: str = "user",
) -> GaussianRouteOverride:
    """Create an explicit override bound to one exact requested route."""

    if allow_nosymm or "nosymm" in gaussian_route_violations(route):
        raise GaussianRoutePolicyError("nosymm is permanently forbidden")

    allowed = frozenset(
        rule
        for rule, enabled in (("tight", allow_tight),)
        if enabled
    )
    override = GaussianRouteOverride(
        route_sha256=gaussian_route_digest(route),
        allowed_rules=allowed,
        reason=str(reason),
        authorized_by=str(authorized_by),
    )
    violations = gaussian_route_violations(route)
    if not violations:
        raise GaussianRoutePolicyError(
            "Gaussian route override was requested for a route with no forbidden rules"
        )
    missing = violations - override.allowed_rules
    if missing:
        raise GaussianRoutePolicyError(
            f"Gaussian route override does not authorize: {', '.join(sorted(missing))}"
        )
    return override


def validate_gaussian_route_policy(
    route: str,
    *,
    override: GaussianRouteOverride | None = None,
) -> frozenset[str]:
    """Validate one requested or externally supplied Gaussian route."""

    violations = gaussian_route_violations(route)
    if "nosymm" in violations:
        raise GaussianRoutePolicyError("nosymm is permanently forbidden")
    if not violations:
        return violations
    if override is None:
        raise GaussianRoutePolicyError(_violation_message(violations))
    if override.route_sha256 != gaussian_route_digest(route):
        raise GaussianRoutePolicyError(
            "Gaussian route override does not match the exact route being validated"
        )
    missing = violations - override.allowed_rules
    if missing:
        raise GaussianRoutePolicyError(_violation_message(missing))
    return violations


def validate_gaussian_route_transformation(
    requested_route: str,
    generated_route: str,
    *,
    override: GaussianRouteOverride | None = None,
) -> None:
    """Validate a route transformation without permitting new violations."""

    approved = validate_gaussian_route_policy(requested_route, override=override)
    generated = gaussian_route_violations(generated_route)
    unexpected = generated - approved
    if unexpected:
        raise GaussianRoutePolicyError(
            "Gaussian route normalization introduced forbidden rules: "
            + ", ".join(sorted(unexpected))
        )


def _violation_message(rules: frozenset[str] | set[str]) -> str:
    labels = ", ".join(sorted(rules))
    return (
        f"Gaussian route violates frozen MATRIX policy ({labels}); NoSymm is "
        "permanently forbidden, and Tight requires exact-route authorization"
    )
