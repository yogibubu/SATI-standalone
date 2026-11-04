"""Typed provenance shared by producers and consumers of electronic observables."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


ELECTRONIC_OBSERVABLE_PROVENANCE_SCHEMA = (
    "matrix.electronic_observable_provenance.v1"
)


class ElectronicObservableOwner(StrEnum):
    """Canonical owners of electronic observables in MATRIX."""

    APOC = "APOC"
    ORACLE = "ORACLE"
    USER = "USER"
    ZAFF = "ZAFF"


class ElectronicObservableKind(StrEnum):
    """Electronic quantities accepted by the consolidated ZAFF seed contract."""

    CM5 = "CM5"
    MAYER = "MAYER"
    ELECTRONEGATIVITY_ESTIMATE = "ELECTRONEGATIVITY_ESTIMATE"
    PAULING_ESTIMATE = "PAULING_ESTIMATE"


class ElectronicObservableCompleteness(StrEnum):
    """Whether a complete vector is primary data or a complete fallback."""

    COMPLETE = "COMPLETE"
    FALLBACK_COMPLETE = "FALLBACK_COMPLETE"


@dataclass(frozen=True)
class ElectronicObservableProvenance:
    """Owner, kind and completeness of one indivisible observable vector."""

    owner: ElectronicObservableOwner
    observable: ElectronicObservableKind
    completeness: ElectronicObservableCompleteness
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner", ElectronicObservableOwner(self.owner))
        object.__setattr__(self, "observable", ElectronicObservableKind(self.observable))
        object.__setattr__(
            self,
            "completeness",
            ElectronicObservableCompleteness(self.completeness),
        )
        normalized_source = str(self.source).strip()
        if not normalized_source:
            raise ValueError("electronic-observable provenance requires a source")
        object.__setattr__(self, "source", normalized_source)

    def to_record(self) -> dict[str, str]:
        """Return the stable JSON record used by MATRIX artifacts."""

        return {
            "owner": self.owner.value,
            "observable": self.observable.value,
            "completeness": self.completeness.value,
            "source": self.source,
        }
