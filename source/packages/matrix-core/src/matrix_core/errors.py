from __future__ import annotations


class OracleError(RuntimeError):
    """Base class for user-facing ORACLE errors."""


class InputError(OracleError):
    """Invalid, missing or inconsistent user input."""


class BackendError(OracleError):
    """External backend failure or unavailable executable."""


class ParseError(OracleError):
    """Input/output parsing failed."""


class ScientificValidationError(OracleError):
    """Scientific contract or consistency check failed."""
