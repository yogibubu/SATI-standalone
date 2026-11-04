"""Small canonical facade for the stable ORACLE perception operations.

The historical exports remain available from :mod:`matrix_oracle`; this module
provides a compact import surface for new integrations.
"""
from __future__ import annotations

from .api import (
    OracleAnalysis,
    OracleAnalysisRequest,
    analyze_structure,
    analyze_structures,
    write_oracle_analysis_reports,
)

__all__ = [
    "OracleAnalysis", "OracleAnalysisRequest", "analyze_structure",
    "analyze_structures", "write_oracle_analysis_reports",
]
