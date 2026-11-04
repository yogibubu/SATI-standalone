"""PySCF launch and structured-output adapters for MATRIX."""

from .dispersion import PySCFDispersionContract, resolve_pyscf_dispersion_contract
from .jobs import PYSCF_EXECUTABLE, PYSCF_SPEC, pyscf_job_status, run_pyscf_job
from .parsers import (
    PYSCF_OUTPUT_FORMAT,
    PYSCF_RESULT_PREFIX,
    PySCFOutputSummary,
    PySCFPointResult,
    hessian_input_from_pyscf_output,
    population_observables_from_pyscf_output,
    read_pyscf_output_geometry,
    read_pyscf_point_result,
    summarize_pyscf_output,
    write_pyscf_hessian_input,
    write_pyscf_point_input,
)
from .population import (
    load_molden_wavefunction,
    population_observables_from_molden,
    population_observables_from_pyscf,
    spin_population_observables_from_pyscf,
)
from .resident import PySCFResidentCalculator, PySCFResidentEvaluation
from .hessian import PySCFHessianEvaluation, pyscf_cartesian_hessian

__all__ = [
    "PYSCF_EXECUTABLE",
    "PYSCF_OUTPUT_FORMAT",
    "PYSCF_RESULT_PREFIX",
    "PYSCF_SPEC",
    "PySCFDispersionContract",
    "PySCFOutputSummary",
    "PySCFPointResult",
    "PySCFResidentCalculator",
    "PySCFResidentEvaluation",
    "PySCFHessianEvaluation",
    "hessian_input_from_pyscf_output",
    "load_molden_wavefunction",
    "population_observables_from_pyscf_output",
    "pyscf_job_status",
    "pyscf_cartesian_hessian",
    "population_observables_from_molden",
    "population_observables_from_pyscf",
    "spin_population_observables_from_pyscf",
    "read_pyscf_output_geometry",
    "read_pyscf_point_result",
    "run_pyscf_job",
    "resolve_pyscf_dispersion_contract",
    "summarize_pyscf_output",
    "write_pyscf_hessian_input",
    "write_pyscf_point_input",
]
