from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkflowStatus(str, Enum):
    MISSING = "missing"
    READY = "ready"
    WARNING = "warning"
    COMPLETE = "complete"


@dataclass(frozen=True)
class WorkflowActionSpec:
    key: str
    label: str
    command: str
    required_sections: tuple[str, ...] = ()
    produced_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowSpec:
    key: str
    title: str
    description: str
    category: str = "workflow"
    required_sections: tuple[str, ...] = ()
    produced_sections: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    publication_exports: tuple[str, ...] = ()
    external_viewers: tuple[str, ...] = ()
    actions: tuple[WorkflowActionSpec, ...] = ()


ORACLE_GUI_WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(
        key="dashboard",
        title="ORACLE Project Dashboard",
        description="Project state, molecule preview, xyzin sections and workflow status.",
        category="project",
        capabilities=(
            "open/create MATRIX projects",
            "inspect xyzin sections",
            "show workflow readiness",
        ),
        actions=(
            WorkflowActionSpec(
                key="validate",
                label="Validate molecule",
                command="validate",
            ),
        ),
    ),
    WindowSpec(
        key="link",
        title="ORACLE Import / Preprocessing",
        description="Import external geometries and materialize the shared enriched XYZ state.",
        category="project",
        produced_sections=(
            "SOURCE",
            "BASIC",
            "SYMMETRY",
            "TOPOLOGY",
            "SYNTHONS",
            "PRIMITIVES",
        ),
        capabilities=(
            "import XYZ, Z-matrix, QM geometry and internal SWITCH SMILES sources",
            "run symmetry detection once with explicit thresholds",
            "materialize topology, synthons and the primitive/B source for downstream tools",
        ),
        actions=(
            WorkflowActionSpec(
                key="preprocess",
                label="Import geometry",
                command="link preprocess",
            ),
        ),
    ),
    WindowSpec(
        key="avogadro",
        title="Molecule Editor / Avogadro Bridge",
        description="Open the XYZ block in Avogadro and import edited coordinates.",
        category="structure",
        required_sections=("SOURCE",),
        external_viewers=("Avogadro", "Avogadro2"),
        capabilities=(
            "view/edit the first XYZ block externally",
            "reimport edited Cartesian coordinates",
            "invalidate or refresh dependent xyzin sections after geometry edits",
        ),
        actions=(
            WorkflowActionSpec(
                key="replace_xyz",
                label="Import edited XYZ block",
                command="replace xyz block",
            ),
        ),
    ),
    WindowSpec(
        key="topology",
        title="Molecular Structure / Synthons",
        description="Inspect molecular structure, synthons, topology and fragment libraries.",
        category="structure",
        required_sections=("TOPOLOGY", "SYNTHONS"),
        produced_sections=("FRAGMENTS",),
        capabilities=(
            "inspect bonds, rings, charges and bond orders",
            "compare and edit synthon classes",
            "fragment molecules and build reusable fragment libraries",
            "prepare nano-lego fragment assembly workflows",
        ),
        publication_exports=("structure tables", "synthon tables", "fragment maps"),
        external_viewers=("Avogadro", "Avogadro2"),
        actions=(
            WorkflowActionSpec(
                key="fragment_plan",
                label="Plan fragments",
                command="fragments plan",
                required_sections=("TOPOLOGY", "SYNTHONS"),
                produced_sections=("FRAGMENTS",),
            ),
            WorkflowActionSpec(
                key="fragment_build",
                label="Build fragments",
                command="fragments build",
                required_sections=("TOPOLOGY", "SYNTHONS"),
                produced_sections=("FRAGMENTS",),
            ),
        ),
    ),
    WindowSpec(
        key="gicforge",
        title="GICForge",
        description="Build, symmetrize and diagnose frozen GICs and B matrices.",
        category="project",
        required_sections=("SYMMETRY", "TOPOLOGY", "SYNTHONS", "PRIMITIVES"),
        produced_sections=("GIC", "SYCART"),
        capabilities=(
            "consume ORACLE primitives and build special/composite SONIC coordinates",
            "symmetrize GICs without mixing coordinate families",
            "evaluate B matrices on the current geometry",
            "write Gaussian input using the frozen GIC contract",
        ),
        actions=(
            WorkflowActionSpec(
                key="gic_build",
                label="Build GICs",
                command="gicforge build",
                required_sections=("SYMMETRY", "TOPOLOGY", "SYNTHONS", "PRIMITIVES"),
                produced_sections=("GIC",),
            ),
            WorkflowActionSpec(
                key="gic_bmatrix",
                label="Evaluate B matrix",
                command="gicforge bmatrix",
                required_sections=("GIC",),
            ),
            WorkflowActionSpec(
                key="gic_gaussian_input",
                label="Write Gaussian input",
                command="gicforge gaussian-input",
                required_sections=("GIC",),
            ),
        ),
    ),
    WindowSpec(
        key="gf",
        title="TRINITY harmonic",
        description="Run harmonic TRINITY from a Cartesian Hessian and frozen GICs.",
        category="vibrational",
        required_sections=("GIC", "CARTESIAN_HESSIAN"),
        produced_sections=("GF_PED",),
        capabilities=(
            "solve harmonic vibrational TRINITY models",
            "scale internal force constants",
            "apply local force-field filtering and nonbonded subtractions",
            "export mode/PED diagnostics for the vibrational workbench",
        ),
        actions=(
            WorkflowActionSpec(
                key="gf_run",
                label="Run TRINITY harmonic",
                command="gf",
                required_sections=("GIC", "CARTESIAN_HESSIAN"),
                produced_sections=("GF_PED",),
            ),
        ),
    ),
    WindowSpec(
        key="sefit",
        title="SEFit / MORPHEUS",
        description="Fit semiexperimental structures and multi-molecule MORPHEUS models.",
        category="rotational",
        required_sections=("ISOTOPOLOGUES",),
        produced_sections=("MORPHEUS",),
        capabilities=(
            "fit semiexperimental structures from rotational constants",
            "compare observed, corrected and calculated rotational constants",
            "run ensemble/multistructure MORPHEUS refinements",
        ),
        actions=(
            WorkflowActionSpec(
                key="semiexp_fit",
                label="Run SEFit",
                command="semiexp",
                required_sections=("ISOTOPOLOGUES",),
                produced_sections=("MORPHEUS",),
            ),
            WorkflowActionSpec(
                key="semiexp_benchmark",
                label="Run paper benchmark",
                command="semiexp-benchmark",
            ),
        ),
    ),
    WindowSpec(
        key="link",
        title="LINK Geometry Optimization",
        description="Prepare external energy/gradient geometry optimizations from xyzin state.",
        category="structure",
        required_sections=("BASIC",),
        produced_sections=("LINK",),
        capabilities=(
            "store a reusable optimization request in #LINK",
            "call an external energy/gradient engine at each future optimization step",
            "use GIC total-symmetric or Cartesian active spaces without reparsing tool inputs",
            "track trajectory, final geometry and energy/gradient logs through the shared xyzin",
        ),
        actions=(
            WorkflowActionSpec(
                key="link_prepare",
                label="Prepare LINK",
                command="link prepare",
                required_sections=("BASIC",),
                produced_sections=("LINK",),
            ),
            WorkflowActionSpec(
                key="link_status",
                label="Summarize LINK",
                command="link status",
                required_sections=("LINK",),
            ),
        ),
    ),
    WindowSpec(
        key="anharmonic",
        title="Anharmonic: VPT2 / VCI / DVR",
        description="Prepare, run and collect VPT2/VCI and DVR workflow state.",
        category="vibrational",
        produced_sections=("VPT2_VCI", "DVR"),
        capabilities=(
            "run VPT2/VCI from normalized QFF data",
            "run DVR directly or collect post-run output",
            "compare anharmonic origins for spectral assignment",
        ),
        actions=(
            WorkflowActionSpec(
                key="vpt2_vci_run",
                label="Run VPT2/VCI",
                command="vpt2-vci",
                required_sections=("QFF",),
                produced_sections=("VPT2_VCI",),
            ),
            WorkflowActionSpec(
                key="dvr_run",
                label="Run DVR",
                command="dvr run",
                required_sections=("DVR",),
                produced_sections=("DVR",),
            ),
        ),
    ),
    WindowSpec(
        key="qm_jobs",
        title="QM Jobs",
        description="Generate inputs, monitor external jobs and normalize QM outputs.",
        category="project",
        produced_sections=(
            "CARTESIAN_HESSIAN",
            "NORMAL_MODES",
            "QFF",
            "VIBRATIONAL",
            "ROTATIONAL",
            "DELTABVIB",
            "ELECTRONIC",
            "TRANSITIONS",
            "ORBITALS",
            "PROPERTIES",
        ),
        capabilities=(
            "write Gaussian GIC inputs from the frozen #GIC contract",
            "inspect and launch Gaussian jobs through the shared job adapter",
            "run formchk on Gaussian checkpoint files",
            "promote FCHK Hessian, normal-mode and QFF data",
            "promote rovibrational output sections from Gaussian logs",
            "promote electronic states, transitions and orbital file records from Gaussian outputs",
            "promote QM properties with explicit unit and conversion metadata",
            "promote Molpro and MRCC geometries through the shared QM adapters",
        ),
        external_viewers=("Avogadro", "Molden"),
        actions=(
            WorkflowActionSpec(
                key="gaussian_from_gic",
                label="Gaussian from GICs",
                command="gicforge gaussian-input",
                required_sections=("GIC",),
            ),
            WorkflowActionSpec(
                key="gaussian_status",
                label="Inspect Gaussian job",
                command="gaussian status",
            ),
            WorkflowActionSpec(
                key="gaussian_run",
                label="Run Gaussian job",
                command="gaussian run",
            ),
            WorkflowActionSpec(
                key="gaussian_formchk",
                label="Run formchk",
                command="gaussian formchk",
            ),
            WorkflowActionSpec(
                key="gaussian_promote_fchk",
                label="Promote Gaussian FCHK",
                command="gaussian promote-fchk",
                produced_sections=(
                    "CARTESIAN_HESSIAN",
                    "NORMAL_MODES",
                    "QFF",
                    "ELECTRONIC",
                    "ORBITALS",
                ),
            ),
            WorkflowActionSpec(
                key="gaussian_promote_electronic",
                label="Promote Gaussian electronic data",
                command="gaussian promote-electronic",
                produced_sections=("ELECTRONIC", "TRANSITIONS", "ORBITALS"),
            ),
            WorkflowActionSpec(
                key="gaussian_promote_rovib",
                label="Promote Gaussian rovib",
                command="gaussian promote-rovib",
                produced_sections=("VIBRATIONAL", "ROTATIONAL", "DELTABVIB"),
            ),
            WorkflowActionSpec(
                key="molpro_promote",
                label="Promote Molpro output",
                command="molpro promote",
                produced_sections=(
                    "SOURCE",
                    "BASIC",
                    "SYMMETRY",
                    "TOPOLOGY",
                    "SYNTHONS",
                    "PRIMITIVES",
                ),
            ),
            WorkflowActionSpec(
                key="mrcc_promote",
                label="Promote MRCC output",
                command="mrcc promote",
                produced_sections=(
                    "SOURCE",
                    "BASIC",
                    "SYMMETRY",
                    "TOPOLOGY",
                    "SYNTHONS",
                    "PRIMITIVES",
                ),
            ),
        ),
    ),
    WindowSpec(
        key="diagnostics",
        title="Diagnostics / Regression",
        description="Run numerical audits against corpus, Fortran77 and benchmark fixtures.",
        category="diagnostics",
        capabilities=(
            "audit GICForge Python/Fortran equivalence",
            "audit demanding GIC regression corpora",
            "run paper benchmark commands",
        ),
        actions=(
            WorkflowActionSpec(
                key="gic_fortran_audit",
                label="GICForge Python/Fortran audit",
                command="gicforge fortran-audit",
            ),
            WorkflowActionSpec(
                key="gic_corpus_audit",
                label="GIC regression corpus audit",
                command="gicforge corpus-audit",
            ),
        ),
    ),
    WindowSpec(
        key="rotational_spectroscopy",
        title="Rotational Spectroscopy",
        description="Collect rotational calculations, inspect assignments and draw publication spectra.",
        category="spectroscopy",
        required_sections=("ROTATIONAL",),
        produced_sections=("ROTATIONAL_SPECTRUM",),
        capabilities=(
            "collect rotational constants and rovibrational corrections",
            "compare isotopologues and semiexperimental residuals",
            "simulate rotational stick/envelope spectra with the local WMS-Rot engine",
            "export WMS-Rot input as a reference/compatibility bridge",
            "scale or select rotational components for planar/asymmetric systems",
        ),
        publication_exports=("CSV line list", "SVG spectrum", "PDF spectrum", "LaTeX table"),
        external_viewers=("WMS-Rot browser reference",),
        actions=(
            WorkflowActionSpec(
                key="rotational_analysis",
                label="Compute rotational state",
                command="rovib rotational",
                required_sections=("BASIC",),
                produced_sections=("ROTATIONAL", "CORIOLIS", "QCENT"),
            ),
            WorkflowActionSpec(
                key="rotational_summary",
                label="Summarize rotational state",
                command="rovib summarize",
                required_sections=("ROTATIONAL",),
            ),
            WorkflowActionSpec(
                key="wmsrot_input",
                label="Export WMS-Rot input",
                command="lock rotor-input",
                required_sections=("ROTATIONAL",),
            ),
            WorkflowActionSpec(
                key="wmsrot_run",
                label="Run LOCK rotational spectrum",
                command="lock rotor-spectrum",
                required_sections=("ROTATIONAL",),
                produced_sections=("ROTATIONAL_SPECTRUM",),
            ),
            WorkflowActionSpec(
                key="open_wmsrot",
                label="Open WMS-Rot",
                command="wmsrot",
            ),
            WorkflowActionSpec(
                key="semiexp_fit",
                label="Run SEFit",
                command="semiexp",
                required_sections=("ISOTOPOLOGUES",),
                produced_sections=("MORPHEUS",),
            ),
        ),
    ),
    WindowSpec(
        key="vibrational_spectroscopy",
        title="Vibrational Spectroscopy",
        description=(
            "Collect harmonic/anharmonic modes, draw spectra and compare gas-phase experiments."
        ),
        category="spectroscopy",
        required_sections=("VIBRATIONAL",),
        produced_sections=("GF_PED", "VPT2_VCI", "DVR", "VIBRATIONAL_SPECTRUM"),
        capabilities=(
            "compare normal modes from GF, Gaussian, VPT2/VCI and DVR",
            "draw heat maps of normal-mode overlaps or contributions",
            "scale force constants and compare frequency shifts",
            "draw harmonic and anharmonic IR, Raman, VCD and ROA spectra when data are present",
            "compare IR/Raman spectra with the second trace mirrored below zero",
            "compare spectra from two calculation levels stored in separate xyzin files",
            "build hybrid frequencies as harmonic(level1)+[anharmonic-harmonic](level2) after normal-mode overlap matching",
            "compare VCD/ROA spectra without mirroring because signed intensities carry information",
            "overlay broadened theoretical spectra with experimental gas-phase NIST IR data",
            "request user instructions when NIST has only condensed-phase data or no spectrum",
        ),
        publication_exports=(
            "CSV peak table",
            "CSV spectrum",
            "SVG spectrum",
            "PDF spectrum",
            "mode heat-map",
        ),
        actions=(
            WorkflowActionSpec(
                key="vibrational_spectrum",
                label="Draw vibrational spectrum",
                command="rovib vib-spectrum",
                required_sections=("VIBRATIONAL",),
                produced_sections=("VIBRATIONAL_SPECTRUM",),
            ),
            WorkflowActionSpec(
                key="vibrational_compare",
                label="Compare vibrational spectra",
                command="rovib vib-compare",
                required_sections=("VIBRATIONAL",),
                produced_sections=("VIBRATIONAL_SPECTRUM",),
            ),
            WorkflowActionSpec(
                key="nist_ir",
                label="Download NIST gas IR",
                command="rovib nist-ir",
            ),
            WorkflowActionSpec(
                key="vibrational_dos",
                label="Build vibrational DOS",
                command="rovib dos",
                required_sections=("VIBRATIONAL",),
            ),
            WorkflowActionSpec(
                key="gf_run",
                label="Run TRINITY harmonic",
                command="gf",
                required_sections=("GIC", "CARTESIAN_HESSIAN"),
                produced_sections=("GF_PED",),
            ),
            WorkflowActionSpec(
                key="vpt2_vci_run",
                label="Run VPT2/VCI",
                command="vpt2-vci",
                required_sections=("QFF",),
                produced_sections=("VPT2_VCI",),
            ),
        ),
    ),
    WindowSpec(
        key="electronic_spectroscopy",
        title="Electronic Spectroscopy",
        description="Collect electronic-state data, visualize orbitals and prepare spectra.",
        category="spectroscopy",
        produced_sections=("ELECTRONIC", "TRANSITIONS", "ORBITALS"),
        capabilities=(
            "collect electronic-state and transition data from QM adapters",
            "visualize orbitals and densities with Molden, Avogadro or MOrbVis",
            "draw UV/visible or electronic stick/broadened spectra",
            "export publication-ready electronic spectra",
        ),
        publication_exports=("CSV transition table", "SVG spectrum", "PDF spectrum"),
        external_viewers=("Avogadro", "Avogadro2", "Molden", "MOrbVis"),
        actions=(
            WorkflowActionSpec(
                key="promote_fchk",
                label="Promote FCHK data",
                command="gaussian promote-fchk",
                produced_sections=("CARTESIAN_HESSIAN", "NORMAL_MODES", "QFF"),
            ),
            WorkflowActionSpec(
                key="view_molden",
                label="View orbitals in Molden",
                command="molden",
            ),
            WorkflowActionSpec(
                key="view_avogadro",
                label="View orbitals in Avogadro",
                command="avogadro2",
            ),
            WorkflowActionSpec(
                key="view_morbvis",
                label="View orbitals in MOrbVis",
                command="morbvis",
            ),
        ),
    ),
    WindowSpec(
        key="thermochemistry_kinetics",
        title="Thermochemistry / Kinetics",
        description="Collect thermal functions, kinetic models and publication tables.",
        category="thermochemistry",
        required_sections=("BASIC", "ROTATIONAL"),
        produced_sections=("THERMO", "KINETICS"),
        capabilities=(
            "summarize rotational and vibrational xyzin sections",
            "run thermochemistry from normalized ORACLE sections",
            "combine vibrational and rovibrational density of states",
            "prepare kinetic-model inputs and compare rates",
            "export publication thermochemistry and kinetics tables",
        ),
        publication_exports=("CSV thermo table", "LaTeX thermo table", "SVG plots", "PDF plots"),
        actions=(
            WorkflowActionSpec(
                key="rovib_summary",
                label="Summarize rovibrational state",
                command="rovib summarize",
                required_sections=("ROTATIONAL",),
            ),
            WorkflowActionSpec(
                key="thermo_run",
                label="Run Thermo",
                command="thermo",
                required_sections=("BASIC", "ROTATIONAL"),
                produced_sections=("THERMO",),
            ),
            WorkflowActionSpec(
                key="vibrational_dos",
                label="Build vibrational DOS",
                command="rovib dos",
                required_sections=("VIBRATIONAL",),
            ),
            WorkflowActionSpec(
                key="rovib_dos",
                label="Build rovibrational DOS",
                command="rovib dos-rovib",
                required_sections=("ROTATIONAL", "VIBRATIONAL"),
            ),
            WorkflowActionSpec(
                key="single_reaction_kinetics",
                label="Run single-reaction TST/RRKM",
                command="kinetics single",
                required_sections=("VIBRATIONAL",),
                produced_sections=("KINETICS",),
            ),
        ),
    ),
)


WINDOWS_BY_KEY: dict[str, WindowSpec] = {spec.key: spec for spec in ORACLE_GUI_WINDOWS}


def window_spec(key: str) -> WindowSpec:
    try:
        return WINDOWS_BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"unknown ORACLE GUI window: {key}") from exc


def all_known_sections() -> tuple[str, ...]:
    sections: set[str] = {"PRIMITIVES"}
    for spec in ORACLE_GUI_WINDOWS:
        sections.update(spec.required_sections)
        sections.update(spec.produced_sections)
        for action in spec.actions:
            sections.update(action.required_sections)
            sections.update(action.produced_sections)
    return tuple(sorted(sections))
