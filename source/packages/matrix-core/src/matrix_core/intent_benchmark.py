"""Reproducible language-contract benchmark for Keymaker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

from .intent import (
    DeterministicIntentCompiler,
    DeterministicLanguageIntentCompiler,
    IntentCompiler,
    IntentRequest,
)


@dataclass(frozen=True)
class IntentBenchmarkCase:
    id: str
    category: str
    request: IntentRequest
    expected_status: str
    expected_recipe: str

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "IntentBenchmarkCase":
        raw = data.get("request", {})
        if not isinstance(raw, Mapping):
            raise ValueError("benchmark request must be an object")
        return cls(
            id=str(data["id"]),
            category=str(data["category"]),
            request=IntentRequest(
                objective=str(raw.get("objective", "")),
                desired_result=str(raw.get("desired_result", "")),
                starting_point=str(raw.get("starting_point", "")),
                validation=str(raw.get("validation", "")),
            ),
            expected_status=str(data["expected_status"]),
            expected_recipe=str(data.get("expected_recipe", "")),
        )


@dataclass(frozen=True)
class IntentBenchmarkRecord:
    id: str
    category: str
    status_match: bool
    recipe_match: bool
    execution_safe: bool
    actual_status: str
    actual_recipe: str

    @property
    def passed(self) -> bool:
        return self.status_match and self.recipe_match and self.execution_safe


@dataclass(frozen=True)
class IntentBenchmarkReport:
    records: tuple[IntentBenchmarkRecord, ...]

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def passed(self) -> int:
        return sum(record.passed for record in self.records)

    @property
    def accuracy(self) -> float:
        return 1.0 if not self.records else self.passed / self.total


@dataclass(frozen=True)
class LanguageIntentBenchmarkCase:
    id: str
    category: str
    text: str
    expected_status: str
    expected_recipe: str
    expected_requirements: Mapping[str, object]


@dataclass(frozen=True)
class LanguageIntentBenchmarkRecord:
    id: str
    category: str
    status_match: bool
    recipe_match: bool
    requirements_match: bool
    execution_safe: bool

    @property
    def passed(self) -> bool:
        return all(
            (self.status_match, self.recipe_match, self.requirements_match, self.execution_safe)
        )


@dataclass(frozen=True)
class LanguageIntentBenchmarkReport:
    records: tuple[LanguageIntentBenchmarkRecord, ...]

    @property
    def total(self) -> int:
        return len(self.records)

    @property
    def passed(self) -> int:
        return sum(record.passed for record in self.records)

    @property
    def accuracy(self) -> float:
        return 1.0 if not self.records else self.passed / self.total


def load_intent_benchmark(path: Path | str) -> tuple[IntentBenchmarkCase, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "matrix.keymaker.intent-benchmark.v1":
        raise ValueError("unsupported intent benchmark schema")
    return tuple(IntentBenchmarkCase.from_dict(item) for item in data.get("cases", ()))


def run_intent_benchmark(
    cases: Iterable[IntentBenchmarkCase],
    *,
    compiler: IntentCompiler | None = None,
) -> IntentBenchmarkReport:
    selected = compiler or DeterministicIntentCompiler()
    records: list[IntentBenchmarkRecord] = []
    for case in cases:
        result = selected.compile(case.request)
        records.append(
            IntentBenchmarkRecord(
                id=case.id,
                category=case.category,
                status_match=result.status == case.expected_status,
                recipe_match=result.selected_recipe == case.expected_recipe,
                execution_safe=result.execution_authorized is False,
                actual_status=result.status,
                actual_recipe=result.selected_recipe,
            )
        )
    return IntentBenchmarkReport(tuple(records))


def load_language_intent_benchmark(
    path: Path | str,
) -> tuple[LanguageIntentBenchmarkCase, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "matrix.keymaker.freeform-intent-benchmark.v1":
        raise ValueError("unsupported free-form intent benchmark schema")
    return tuple(
        LanguageIntentBenchmarkCase(
            id=str(item["id"]),
            category=str(item["category"]),
            text=str(item["text"]),
            expected_status=str(item["expected_status"]),
            expected_recipe=str(item.get("expected_recipe", "")),
            expected_requirements=dict(item.get("expected_requirements", {})),
        )
        for item in data.get("cases", ())
    )


def run_language_intent_benchmark(
    cases: Iterable[LanguageIntentBenchmarkCase],
    *,
    compiler: DeterministicLanguageIntentCompiler | None = None,
) -> LanguageIntentBenchmarkReport:
    selected = compiler or DeterministicLanguageIntentCompiler()
    records: list[LanguageIntentBenchmarkRecord] = []
    for case in cases:
        result = selected.compile_text(case.text)
        records.append(
            LanguageIntentBenchmarkRecord(
                id=case.id,
                category=case.category,
                status_match=result.status == case.expected_status,
                recipe_match=result.selected_recipe == case.expected_recipe,
                requirements_match=all(
                    getattr(result.requirements, key, None) == value
                    for key, value in case.expected_requirements.items()
                ),
                execution_safe=result.execution_authorized is False,
            )
        )
    return LanguageIntentBenchmarkReport(tuple(records))
