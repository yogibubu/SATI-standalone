"""Typed intent boundary for Keymaker.

Natural language may propose a workflow, but it never authorizes or executes
one.  This module deliberately depends only on the deterministic workflow
registry and exposes no subprocess, network or GUI operations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Mapping, Protocol

from .atomic_io import atomic_json_write
from .workflow import (
    WorkflowPlan,
    WorkflowRecommendation,
    build_workflow_plan,
    rank_workflow_recipes,
)
from .workspace import WorkspaceLayout, ensure_workspace


INTENT_SCHEMA = "matrix.keymaker.intent.v1"
INTENT_STATE_FILENAME = "keymaker-intent.json"
INTENT_STATUSES = ("ready", "needs_clarification", "unsupported")
DETERMINISTIC_COMPILER_ID = "matrix.deterministic-intent.v1"
DETERMINISTIC_LANGUAGE_COMPILER_ID = "matrix.deterministic-language-intent.v1"
GUARDED_LANGUAGE_COMPILER_ID = "matrix.guarded-language-intent.v1"
DERIVATIVE_POLICIES = ("analytic", "prefer-analytic", "allow-numerical")


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


@dataclass(frozen=True)
class IntentRequest:
    objective: str
    desired_result: str
    starting_point: str
    validation: str

    def normalized(self) -> "IntentRequest":
        return IntentRequest(*(_clean(value) for value in asdict(self).values()))


@dataclass(frozen=True)
class IntentCandidate:
    recipe_key: str
    title: str
    score: int
    confidence: float
    matched_terms: tuple[str, ...]

    @classmethod
    def from_recommendation(cls, value: WorkflowRecommendation) -> "IntentCandidate":
        return cls(
            recipe_key=value.recipe_key,
            title=value.title,
            score=value.score,
            confidence=value.confidence,
            matched_terms=value.matched_terms,
        )


@dataclass(frozen=True)
class ScientificRequirements:
    backend: str = "auto"
    method: str = ""
    basis: str = ""
    derivative: str = ""
    derivative_policy: str = "prefer-analytic"
    require_ecp: bool = False
    execution: str = "auto"
    remote_machine: str = ""

    def __post_init__(self) -> None:
        if self.derivative not in {"", "energy", "gradient", "hessian"}:
            raise ValueError(f"unknown requested derivative: {self.derivative}")
        if self.derivative_policy not in DERIVATIVE_POLICIES:
            raise ValueError(f"unknown derivative policy: {self.derivative_policy}")
        if self.execution not in {"auto", "local", "remote"}:
            raise ValueError(f"unknown requested execution mode: {self.execution}")


@dataclass(frozen=True)
class IntentIssue:
    code: str
    severity: str
    field: str
    message: str

    def __post_init__(self) -> None:
        if self.severity not in {"info", "warning", "error"}:
            raise ValueError(f"unknown intent-issue severity: {self.severity}")


@dataclass(frozen=True)
class IntentCompilation:
    request: IntentRequest
    status: str
    selected_recipe: str
    candidates: tuple[IntentCandidate, ...]
    missing_fields: tuple[str, ...]
    clarification_questions: tuple[str, ...]
    compiler_id: str = DETERMINISTIC_COMPILER_ID
    execution_authorized: bool = False
    requirements: ScientificRequirements = field(default_factory=ScientificRequirements)
    confidence: float = 0.0
    issues: tuple[IntentIssue, ...] = ()
    source_text: str = ""
    proposal_origin: str = "deterministic"
    schema: str = INTENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != INTENT_SCHEMA:
            raise ValueError(f"unsupported intent schema: {self.schema}")
        if self.status not in INTENT_STATUSES:
            raise ValueError(f"unknown intent status: {self.status}")
        if self.execution_authorized:
            raise ValueError("intent compilation cannot authorize execution")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("intent confidence must be between zero and one")
        if self.status == "ready" and not self.selected_recipe:
            raise ValueError("a ready intent requires a selected recipe")
        if self.status != "ready" and self.selected_recipe:
            raise ValueError("an unresolved intent cannot select a recipe")

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "IntentCompilation":
        raw_request = data.get("request", {})
        if not isinstance(raw_request, Mapping):
            raise ValueError("intent request must be an object")
        raw_candidates = data.get("candidates", ())
        if not isinstance(raw_candidates, (list, tuple)):
            raise ValueError("intent candidates must be an array")
        candidates = tuple(
            IntentCandidate(
                recipe_key=str(item["recipe_key"]),
                title=str(item["title"]),
                score=int(item["score"]),
                confidence=float(item["confidence"]),
                matched_terms=tuple(item.get("matched_terms", ())),
            )
            for item in raw_candidates
            if isinstance(item, Mapping)
        )
        raw_requirements = data.get("requirements", {})
        if not isinstance(raw_requirements, Mapping):
            raise ValueError("intent requirements must be an object")
        raw_issues = data.get("issues", ())
        if not isinstance(raw_issues, (list, tuple)):
            raise ValueError("intent issues must be an array")
        return cls(
            request=IntentRequest(
                objective=str(raw_request.get("objective", "")),
                desired_result=str(raw_request.get("desired_result", "")),
                starting_point=str(raw_request.get("starting_point", "")),
                validation=str(raw_request.get("validation", "")),
            ),
            status=str(data.get("status", "")),
            selected_recipe=str(data.get("selected_recipe", "")),
            candidates=candidates,
            missing_fields=tuple(data.get("missing_fields", ())),
            clarification_questions=tuple(data.get("clarification_questions", ())),
            compiler_id=str(data.get("compiler_id", "")),
            execution_authorized=bool(data.get("execution_authorized", False)),
            requirements=ScientificRequirements(
                backend=str(raw_requirements.get("backend", "auto")),
                method=str(raw_requirements.get("method", "")),
                basis=str(raw_requirements.get("basis", "")),
                derivative=str(raw_requirements.get("derivative", "")),
                derivative_policy=str(
                    raw_requirements.get("derivative_policy", "prefer-analytic")
                ),
                require_ecp=bool(raw_requirements.get("require_ecp", False)),
                execution=str(raw_requirements.get("execution", "auto")),
                remote_machine=str(raw_requirements.get("remote_machine", "")),
            ),
            confidence=float(data.get("confidence", 0.0)),
            issues=tuple(
                IntentIssue(
                    code=str(item.get("code", "")),
                    severity=str(item.get("severity", "error")),
                    field=str(item.get("field", "")),
                    message=str(item.get("message", "")),
                )
                for item in raw_issues
                if isinstance(item, Mapping)
            ),
            source_text=str(data.get("source_text", "")),
            proposal_origin=str(data.get("proposal_origin", "deterministic")),
            schema=str(data.get("schema", "")),
        )


class IntentCompiler(Protocol):
    compiler_id: str

    def compile(self, request: IntentRequest) -> IntentCompilation: ...


class DeterministicIntentCompiler:
    """Compile explicit chemical fields without executing or authorizing tools."""

    compiler_id = DETERMINISTIC_COMPILER_ID

    def compile(self, request: IntentRequest) -> IntentCompilation:
        normalized = request.normalized()
        combined_text = " ".join(asdict(normalized).values())
        requirements, issues = _extract_scientific_requirements(combined_text)
        missing = tuple(
            name for name, value in asdict(normalized).items() if not str(value).strip()
        )
        if missing:
            questions = tuple(_missing_field_question(name) for name in missing)
            return IntentCompilation(
                request=normalized,
                status="needs_clarification",
                selected_recipe="",
                candidates=(),
                missing_fields=missing,
                clarification_questions=questions,
                compiler_id=self.compiler_id,
                requirements=requirements,
                issues=issues,
            )

        context = " ".join(
            (normalized.desired_result, normalized.starting_point, normalized.validation)
        )
        ranked = rank_workflow_recipes(normalized.objective, context)
        positive = tuple(
            IntentCandidate.from_recommendation(candidate)
            for candidate in ranked
            if candidate.score > 0
        )
        blocking_issues = tuple(issue for issue in issues if issue.severity == "error")
        if blocking_issues:
            return IntentCompilation(
                request=normalized,
                status="needs_clarification",
                selected_recipe="",
                candidates=positive[:3],
                missing_fields=tuple(sorted({issue.field for issue in blocking_issues})),
                clarification_questions=tuple(issue.message for issue in blocking_issues),
                compiler_id=self.compiler_id,
                requirements=requirements,
                confidence=positive[0].confidence if positive else 0.0,
                issues=issues,
            )
        if not positive:
            return IntentCompilation(
                request=normalized,
                status="unsupported",
                selected_recipe="",
                candidates=(),
                missing_fields=("workflow",),
                clarification_questions=(
                    "Which scientific result should MATRIX produce: coordinates, an optimized "
                    "structure, a PES exploration, frequencies, a force field, a refined "
                    "structure, or a structural correction?",
                ),
                compiler_id=self.compiler_id,
                requirements=requirements,
                issues=issues,
            )

        best_score = positive[0].score
        tied = tuple(candidate for candidate in positive if candidate.score == best_score)
        if len(tied) > 1:
            choices = "; ".join(f"{item.title} ({item.recipe_key})" for item in tied)
            return IntentCompilation(
                request=normalized,
                status="needs_clarification",
                selected_recipe="",
                candidates=tied,
                missing_fields=("workflow",),
                clarification_questions=(
                    f"The request matches more than one workflow: {choices}. "
                    "Which result is primary?",
                ),
                compiler_id=self.compiler_id,
                requirements=requirements,
                confidence=tied[0].confidence,
                issues=issues,
            )

        return IntentCompilation(
            request=normalized,
            status="ready",
            selected_recipe=positive[0].recipe_key,
            candidates=positive[:3],
            missing_fields=(),
            clarification_questions=(),
            compiler_id=self.compiler_id,
            requirements=requirements,
            confidence=positive[0].confidence,
            issues=issues,
        )


class LanguageProposalProvider(Protocol):
    """Optional AI boundary: return data only, never commands or authorization."""

    provider_id: str

    def propose(self, text: str) -> Mapping[str, object]: ...


class DeterministicLanguageIntentCompiler:
    """Compile one natural-language request without requiring an AI service."""

    compiler_id = DETERMINISTIC_LANGUAGE_COMPILER_ID

    def __init__(self, verifier: DeterministicIntentCompiler | None = None) -> None:
        self.verifier = verifier or DeterministicIntentCompiler()

    def compile_text(self, text: str) -> IntentCompilation:
        source = _clean(text)
        if not source:
            return replace(
                self.verifier.compile(IntentRequest("", "", "", "")),
                compiler_id=self.compiler_id,
                source_text="",
                proposal_origin="deterministic-language",
            )
        request = _request_from_language(source)
        result = self.verifier.compile(request)
        return replace(
            result,
            compiler_id=self.compiler_id,
            source_text=source,
            proposal_origin="deterministic-language",
        )


class GuardedLanguageIntentCompiler:
    """Verify an optional AI proposal with the deterministic v1 compiler.

    Provider output is treated as untrusted structured data.  Unknown fields,
    execution requests and malformed proposals are discarded and recorded.
    """

    compiler_id = GUARDED_LANGUAGE_COMPILER_ID

    def __init__(
        self,
        provider: LanguageProposalProvider | None = None,
        *,
        fallback: DeterministicLanguageIntentCompiler | None = None,
        verifier: DeterministicIntentCompiler | None = None,
    ) -> None:
        self.provider = provider
        self.fallback = fallback or DeterministicLanguageIntentCompiler(verifier)
        self.verifier = verifier or self.fallback.verifier

    def compile_text(self, text: str) -> IntentCompilation:
        fallback_result = self.fallback.compile_text(text)
        if self.provider is None:
            return replace(
                fallback_result,
                compiler_id=self.compiler_id,
                proposal_origin="deterministic-fallback",
            )
        try:
            proposal = self.provider.propose(_clean(text))
        except Exception as exc:  # provider failure must not break project creation
            issue = IntentIssue(
                "proposal_provider_failed",
                "warning",
                "language_provider",
                f"The language provider failed; deterministic interpretation was used ({exc}).",
            )
            return replace(
                fallback_result,
                compiler_id=self.compiler_id,
                issues=(*fallback_result.issues, issue),
                proposal_origin="deterministic-fallback",
            )
        if not isinstance(proposal, Mapping):
            issue = IntentIssue(
                "invalid_proposal",
                "warning",
                "language_provider",
                "The language provider returned no structured mapping; deterministic interpretation was used.",
            )
            return replace(
                fallback_result,
                compiler_id=self.compiler_id,
                issues=(*fallback_result.issues, issue),
                proposal_origin="deterministic-fallback",
            )

        issues: list[IntentIssue] = []
        forbidden = {
            "execution_authorized", "execute", "command", "subprocess", "ssh", "approved"
        } & set(proposal)
        if forbidden:
            issues.append(
                IntentIssue(
                    "unsafe_proposal_fields",
                    "warning",
                    "authorization",
                    "The provider attempted to cross the execution boundary; those fields were discarded.",
                )
            )
        raw_request = proposal.get("request", proposal)
        allowed = {"objective", "desired_result", "starting_point", "validation"}
        if not isinstance(raw_request, Mapping):
            raw_request = {}
        unknown = set(raw_request) - allowed
        if unknown:
            issues.append(
                IntentIssue(
                    "unknown_proposal_fields",
                    "warning",
                    "language_provider",
                    f"Unknown proposal fields were discarded: {', '.join(sorted(unknown))}.",
                )
            )
        base_request = fallback_result.request
        values: dict[str, str] = {}
        for name in allowed:
            proposed = raw_request.get(name, "")
            if proposed and not isinstance(proposed, str):
                issues.append(
                    IntentIssue(
                        "non_string_proposal_field",
                        "warning",
                        name,
                        f"The provider's {name} field was not text and was discarded.",
                    )
                )
                proposed = ""
            values[name] = _clean(proposed) or getattr(base_request, name)
        verified = self.verifier.compile(IntentRequest(**values))
        provider_id = str(getattr(self.provider, "provider_id", "external-provider"))
        return replace(
            verified,
            compiler_id=self.compiler_id,
            execution_authorized=False,
            issues=(*verified.issues, *issues),
            source_text=_clean(text),
            proposal_origin=f"ai-proposal-verified:{provider_id}",
        )


_BACKEND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("gdv", r"\bgdv\b|gaussian development"),
    ("g16", r"\bg16\b|gaussian\s*16"),
    ("orca", r"\borca\b"),
    ("nwchem", r"\bnwchem\b"),
    ("molpro", r"\bmolpro\b"),
    ("mrcc", r"\bmrcc\b"),
    ("cfour", r"\b(?:cfour|c4)\b"),
    ("xtb", r"\b(?:x-?tb|gfn[012]-?xtb)\b"),
    ("pyscf", r"\bpyscf\b"),
    ("psi4", r"\bpsi4\b"),
    ("et", r"\beT\b"),
)

_METHOD_PATTERNS: tuple[tuple[str, str], ...] = (
    ("jun-ChS", r"\bjun(?:[- ]?chs|[- ]?cheap)\b"),
    ("PPCS2", r"\bppcs2\b"),
    ("PCS2", r"\bpcs2\b"),
    ("CCSDT(Q)", r"\bccsdt\s*\(q\)\b"),
    ("CCSD(T)", r"\bccsd\s*\(t\)\b"),
    ("DLPNO-CCSD(T)", r"\bdlpno-?ccsd\s*\(t\)\b"),
    ("CCSDT", r"\bccsdt\b"),
    ("CC3", r"\bcc3\b"),
    ("CC2", r"\bcc2\b"),
    ("CCSD", r"\bccsd\b"),
    ("CASSCF", r"\bcasscf\b"),
    ("MP2", r"\b(?:scs-?)?mp2\b"),
    ("PBE0", r"\bpbe0\b"),
    ("B3LYP", r"\bb3lyp\b"),
    ("HF", r"\b(?:r?hf|uhf|rohf|hartree[- ]fock)\b"),
    ("GFN2-XTB", r"\bgfn2-?xtb\b"),
)


def _extract_scientific_requirements(
    text: str,
) -> tuple[ScientificRequirements, tuple[IntentIssue, ...]]:
    source = str(text)
    folded = source.casefold()
    issues: list[IntentIssue] = []

    backends = {
        key
        for key, pattern in _BACKEND_PATTERNS
        if re.search(
            rf"\b(?:using|with|backend|program|code|usando|con|programma|codice)"
            rf"\s*[:=]?\s*(?:il\s+)?(?:{pattern})",
            source,
            re.IGNORECASE,
        )
    }
    if len(backends) > 1:
        issues.append(
            IntentIssue(
                "multiple_backends", "error", "backend",
                f"More than one backend was requested ({', '.join(sorted(backends))}); choose one or use auto.",
            )
        )
    backend = next(iter(backends), "auto") if len(backends) <= 1 else "auto"

    methods = {
        label for label, pattern in _METHOD_PATTERNS if re.search(pattern, source, re.IGNORECASE)
    }
    # Longer coupled-cluster names contain shorter tokens; retain only the
    # most specific textual match.
    for specific, contained in (
        ("CCSDT(Q)", ("CCSDT", "CCSD")),
        ("CCSD(T)", ("CCSD",)),
        ("DLPNO-CCSD(T)", ("CCSD(T)", "CCSD")),
        ("CCSDT", ("CCSD",)),
    ):
        if specific in methods:
            methods.difference_update(contained)
    if len(methods) > 1:
        issues.append(
            IntentIssue(
                "multiple_methods", "error", "method",
                f"More than one electronic-structure method was requested ({', '.join(sorted(methods))}); identify the primary method.",
            )
        )
    method = next(iter(methods), "") if len(methods) <= 1 else ""

    basis = ""
    slash = re.search(
        r"(?:ccsdt?\s*\(t\)|ccsdt|cc3|cc2|ccsd|mp2|pbe0|b3lyp|r?hf|uhf|rohf)\s*/\s*([A-Za-z0-9+*(),._-]+)",
        source,
        re.IGNORECASE,
    )
    named_basis = re.search(
        r"(?:basis|base)\s+(?:set\s+)?(?:atomica\s+)?([A-Za-z0-9+*(),._-]+)",
        source,
        re.IGNORECASE,
    )
    if slash:
        basis = slash.group(1)
    elif named_basis:
        basis = named_basis.group(1)

    analytic = bool(re.search(r"\b(?:analytic(?:al)?|analitic[oa])\b", folded))
    numerical = bool(re.search(r"\b(?:numerical|numeric|numerico|numerica)\b", folded))
    if analytic and numerical:
        issues.append(
            IntentIssue(
                "conflicting_derivative_policy", "error", "derivative_policy",
                "Both analytical and numerical derivatives were requested; choose the required policy.",
            )
        )
    policy = "analytic" if analytic and not numerical else (
        "allow-numerical" if numerical and not analytic else "prefer-analytic"
    )
    if re.search(r"\b(?:hessian\w*|force constants?|frequenc\w*|frequenz\w*|normal modes?|modi normali)\b", folded):
        derivative = "hessian"
    elif re.search(r"\b(?:gradient\w*|optimi[sz]\w*|ottimizz\w*)\b", folded):
        derivative = "gradient"
    elif re.search(r"\b(?:energy|energia|single[- ]point)\b", folded):
        derivative = "energy"
    else:
        derivative = ""

    require_ecp = bool(re.search(r"\b(?:ecp|pseudopotential|pseudopotenziale)\b", folded))
    if backend == "xtb" and basis:
        issues.append(
            IntentIssue(
                "xtb_with_orbital_basis", "error", "basis",
                "xTB uses its fixed parametrization and cannot consume the requested orbital basis.",
            )
        )
    if backend == "et" and require_ecp:
        issues.append(
            IntentIssue(
                "et_external_ecp_unavailable", "error", "backend",
                "The verified eT adapter has no external ECP contract; choose another backend or remove the ECP requirement.",
            )
        )

    remote = re.search(
        r"(?:remote(?:\s+machine)?|remot[oa]|machine|macchina|host)\s+(?:on\s+|su\s+)?([A-Za-z0-9_.@-]+)",
        source,
        re.IGNORECASE,
    )
    execution = "remote" if remote else (
        "local" if re.search(r"\b(?:locally|local|locale)\b", folded) else "auto"
    )
    return (
        ScientificRequirements(
            backend=backend,
            method=method,
            basis=basis,
            derivative=derivative,
            derivative_policy=policy,
            require_ecp=require_ecp,
            execution=execution,
            remote_machine=remote.group(1) if remote else "",
        ),
        tuple(issues),
    )


def _request_from_language(text: str) -> IntentRequest:
    source = _clean(text)
    objective = re.split(
        r"\b(?:and\s+(?:verify|validate|check)|e\s+(?:verifica|controlla|valida))\b",
        source,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,;.")
    lowered = source.casefold()
    desired = ""
    if re.search(r"\b(?:optimi[sz]\w*|ottimizz\w*)\b", lowered):
        desired = "A converged minimum-energy structure"
    elif re.search(r"\b(?:frequenc\w*|frequenz\w*|normal modes?|modi normali|vibrazion\w*)\b", lowered):
        desired = "Harmonic frequencies and normal modes"
    elif re.search(r"\b(?:sonic|internal coordinates?|coordinate interne)\b", lowered):
        desired = "A human-readable nonredundant internal-coordinate definition"
    elif re.search(r"\b(?:force field|campo di forza|zaff)\b", lowered):
        desired = "A validated force field"
    elif re.search(r"\b(?:scan|pes|potential energy surface)\b", lowered):
        desired = "A validated set of potential-energy-surface points"

    file_match = re.search(
        r"(?:from|starting\s+from|partendo\s+da|da)\s+([^\s,;]+\.(?:xyz|sdf|mol2?|pdb|cif|gjf|com|log|fchk|molden|msr))\b",
        source,
        re.IGNORECASE,
    )
    smiles_match = re.search(r"\bsmiles\s*[:=]?\s*([^\s,;]+)", source, re.IGNORECASE)
    starting = (
        f"Structure file {file_match.group(1)}" if file_match else
        (f"SMILES {smiles_match.group(1)}" if smiles_match else "")
    )

    validation_match = re.search(
        r"\b(?:verify|validate|check|verifica|controlla|valida)\b(?P<value>.+)$",
        source,
        re.IGNORECASE,
    )
    validation = _clean(validation_match.group("value").strip(" ,;.")) if validation_match else ""
    return IntentRequest(objective, desired, starting, validation)


class DeterministicPlanAssembler:
    """Turn a resolved intent into a workflow plan without launching it."""

    def build(self, compilation: IntentCompilation, *, project_id: str) -> WorkflowPlan:
        if not compilation.ready:
            raise ValueError("cannot build a workflow plan from an unresolved intent")
        if compilation.execution_authorized:
            raise ValueError("intent compilation must not carry execution authorization")
        return build_workflow_plan(
            compilation.selected_recipe,
            objective=compilation.request.objective,
            project_id=project_id,
        )


@dataclass(frozen=True)
class ResultContext:
    project_id: str
    recipe_key: str
    status: str
    completed_steps: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    failures: tuple[str, ...]


def result_context(plan: WorkflowPlan) -> ResultContext:
    """Expose compact validated state, never raw numerical arrays, to a narrator."""

    completed = tuple(step.id for step in plan.steps if step.status == "completed")
    artifacts = sorted(plan.observed_artifacts)
    failures = tuple(
        f"{step.id}: {step.last_error}" for step in plan.steps if step.last_error
    )
    return ResultContext(
        project_id=plan.project_id,
        recipe_key=plan.recipe_key,
        status=plan.status,
        completed_steps=completed,
        artifact_ids=tuple(artifacts),
        failures=failures,
    )


class DeterministicResultNarrator:
    """Render a safe summary from validated workflow metadata only."""

    def narrate(self, context: ResultContext) -> str:
        if context.status == "completed":
            return (
                f"Project {context.project_id} completed the {context.recipe_key} workflow; "
                f"{len(context.artifact_ids)} validated artifact(s) are recorded."
            )
        if context.failures:
            return f"Project {context.project_id} stopped: {'; '.join(context.failures)}"
        return (
            f"Project {context.project_id} is {context.status}; "
            f"{len(context.completed_steps)} workflow step(s) are complete."
        )


def _missing_field_question(name: str) -> str:
    return {
        "objective": "What chemical question should the project answer?",
        "desired_result": "Which observable or concrete result should be returned?",
        "starting_point": "What molecular structure, SMILES, spectrum, or dataset is available?",
        "validation": "How should the result be validated?",
    }[name]


def intent_schema_path() -> Path:
    return Path(str(files("matrix_core").joinpath("schemas", "intent-v1.schema.json")))


def intent_compilation_path(workspace: Path | WorkspaceLayout) -> Path:
    layout = workspace if isinstance(workspace, WorkspaceLayout) else ensure_workspace(Path(workspace))
    layout.ensure()
    return layout.state / INTENT_STATE_FILENAME


def write_intent_compilation(
    compilation: IntentCompilation,
    workspace: Path | WorkspaceLayout,
) -> Path:
    path = intent_compilation_path(workspace)
    atomic_json_write(path, compilation.to_dict())
    return path


def read_intent_compilation(workspace: Path | WorkspaceLayout) -> IntentCompilation:
    path = intent_compilation_path(workspace)
    return IntentCompilation.from_dict(json.loads(path.read_text(encoding="utf-8")))
