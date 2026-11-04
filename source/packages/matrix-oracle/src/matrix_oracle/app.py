from __future__ import annotations

import argparse
from pathlib import Path

from matrix_core.online_help import online_help_text
from matrix_core import load_environment

from .commands import OracleGuiCommand
from .contracts import (
    ToolContractTable,
    load_tool_contract_gui_state,
    tool_contract_gui_state_lines,
)
from .dashboard import DashboardAction, OracleDashboardController
from .electronic import (
    ElectronicTable,
    MORBVIS_URL,
    OracleElectronicController,
    default_electronic_export_dir,
    electronic_gui_state_lines,
    load_electronic_gui_state,
)
from .gicforge import (
    GICForgeTable,
    OracleGICForgeController,
    default_gicforge_bmatrix_output,
    default_gicforge_gaussian_output,
    default_gicforge_report_output,
    gicforge_gui_state_lines,
    load_gicforge_gui_state,
)
from .gf import (
    GFTable,
    OracleGFController,
    default_gf_csv_dir,
    default_gf_report_output,
    gf_gui_state_lines,
    load_gf_gui_state,
)
from .guidance import missing_sections_message
from .project import load_oracle_project_state
from .qm_jobs import (
    OracleQMJobsController,
    default_qm_formchk_output,
    default_qm_gaussian_input_output,
    default_qm_gaussian_workdir,
)
from .sefit import (
    OracleSEFitController,
    SEFitTable,
    default_sefit_outdir,
    load_sefit_gui_state,
    sefit_gui_state_lines,
)
from .structure import (
    OracleStructureController,
    StructureTable,
    default_preprocess_output,
    load_structure_gui_state,
)
from .thermo_kinetics import (
    OracleThermoKineticsController,
    ThermoKineticsTable,
    default_rovib_q_output,
    default_rovibrational_dos_output,
    default_rotational_dos_output,
    default_thermo_export_dir,
    default_thermo_report_output,
    default_vibrational_dos_output,
    load_thermo_kinetics_gui_state,
    thermo_kinetics_gui_state_lines,
)
from .link import (
    OracleLinkController,
    LinkTable,
    default_link_run_dir,
    load_link_gui_state,
    link_gui_state_lines,
)
from .workbench import (
    WorkbenchTable,
    load_workbench_gui_state,
    workbench_gui_state_lines,
)
from .workflows import ORACLE_GUI_WINDOWS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ORACLE desktop GUI")
    parser.add_argument("xyzin", nargs="?", type=Path, help="MATRIX enriched XYZ project")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run_qt(args.xyzin)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("PySide6"):
            raise SystemExit(
                "PySide6 is not installed. Install the GUI extra with "
                "`pip install -e packages/matrix-oracle[qt]` or run the headless "
                "matrix_oracle project controllers from Python."
            ) from exc
        raise


def _run_qt(initial_xyzin: Path | None) -> int:
    from PySide6.QtCore import QProcess, Qt
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QHBoxLayout,
        QCheckBox,
        QLabel,
        QComboBox,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    class OracleDashboardWindow(QMainWindow):
        def __init__(self, xyzin: Path | None = None) -> None:
            super().__init__()
            self.controller = OracleDashboardController(xyzin)
            self.structure_controller = OracleStructureController(xyzin)
            self.gicforge_controller = OracleGICForgeController(xyzin)
            self.gf_controller = OracleGFController(xyzin)
            self.qm_jobs_controller = OracleQMJobsController(xyzin)
            self.electronic_controller = OracleElectronicController(xyzin)
            self.sefit_controller = OracleSEFitController(xyzin)
            self.thermo_kinetics_controller = OracleThermoKineticsController(xyzin)
            self.link_controller = OracleLinkController(xyzin)
            self.process: QProcess | None = None
            self.current_actions: tuple[DashboardAction, ...] = ()
            self.pending_xyzin_after_run: Path | None = None
            self.workbench_tabs = (
                ("anharmonic", "Anharmonic"),
                ("diagnostics", "Diagnostics"),
                ("rotational_spectroscopy", "Rotational"),
                ("vibrational_spectroscopy", "Vibrational"),
            )
            self.workbench_summaries: dict[str, QTextEdit] = {}
            self.workbench_section_tables: dict[str, QTableWidget] = {}
            self.workbench_action_tables: dict[str, QTableWidget] = {}
            self.workbench_capability_tables: dict[str, QTableWidget] = {}
            self.workbench_export_tables: dict[str, QTableWidget] = {}
            self.setWindowTitle("ORACLE Project Dashboard")
            self.resize(1280, 780)

            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)

            toolbar = QHBoxLayout()
            self.path_label = QLabel("No MATRIX project loaded")
            self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            open_button = QPushButton("Open")
            open_button.clicked.connect(self.open_project)
            refresh_button = QPushButton("Refresh")
            refresh_button.clicked.connect(self.refresh)
            self.run_button = QPushButton("Run Selected")
            self.run_button.clicked.connect(self.run_selected_action)
            toolbar.addWidget(self.path_label, stretch=1)
            toolbar.addWidget(open_button)
            toolbar.addWidget(refresh_button)
            toolbar.addWidget(self.run_button)
            layout.addLayout(toolbar)

            tabs = QTabWidget()
            state_tab = QWidget()
            body = QHBoxLayout(state_tab)
            self.workflow_list = QListWidget()
            self.workflow_list.itemDoubleClicked.connect(self.show_workflow_details)
            self.section_table = QTableWidget(0, 3)
            self.section_table.setHorizontalHeaderLabels(("Section", "Lines", "Schema"))
            body.addWidget(self.workflow_list, stretch=1)
            body.addWidget(self.section_table, stretch=2)
            tabs.addTab(state_tab, "Project State")

            actions_tab = QWidget()
            actions_layout = QVBoxLayout(actions_tab)
            self.action_table = QTableWidget(0, 4)
            self.action_table.setHorizontalHeaderLabels(("Action", "Status", "Window", "Command"))
            self.action_table.itemDoubleClicked.connect(self.run_selected_action)
            actions_layout.addWidget(self.action_table)
            tabs.addTab(actions_tab, "Actions")
            contracts_tab = self._build_contracts_tab()
            tabs.addTab(contracts_tab, "Tool Contracts")
            help_tab = self._build_help_tab()
            tabs.addTab(help_tab, "Help")
            structure_tab = self._build_structure_tab()
            tabs.addTab(structure_tab, "Structure")
            gicforge_tab = self._build_gicforge_tab()
            tabs.addTab(gicforge_tab, "GICForge")
            gf_tab = self._build_gf_tab()
            tabs.addTab(gf_tab, "TRINITY")
            qm_jobs_tab = self._build_qm_jobs_tab()
            tabs.addTab(qm_jobs_tab, "QM Jobs")
            electronic_tab = self._build_electronic_tab()
            tabs.addTab(electronic_tab, "Electronic")
            sefit_tab = self._build_sefit_tab()
            tabs.addTab(sefit_tab, "SEFit")
            thermo_kinetics_tab = self._build_thermo_kinetics_tab()
            tabs.addTab(thermo_kinetics_tab, "Thermo/Kinetics")
            link_tab = self._build_link_tab()
            tabs.addTab(link_tab, "LINK")
            for key, label in self.workbench_tabs:
                tabs.addTab(self._build_workbench_tab(key), label)
            layout.addWidget(tabs, stretch=2)

            self.details = QTextEdit()
            self.details.setReadOnly(True)
            self.log_output = QTextEdit()
            self.log_output.setReadOnly(True)

            bottom_tabs = QTabWidget()
            bottom_tabs.addTab(self.details, "Details")
            bottom_tabs.addTab(self.log_output, "Log")
            layout.addWidget(bottom_tabs, stretch=1)

            self.refresh()

        def open_project(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Open MATRIX xyzin",
                str(Path.cwd()),
                "MATRIX xyzin (*.xyzin xyzin *.xyz);;All files (*)",
            )
            if path:
                self.controller.set_xyzin(Path(path))
                self.structure_controller.set_xyzin(Path(path))
                self.gicforge_controller.set_xyzin(Path(path))
                self.gf_controller.set_xyzin(Path(path))
                self.qm_jobs_controller.set_xyzin(Path(path))
                self.electronic_controller.set_xyzin(Path(path))
                self.sefit_controller.set_xyzin(Path(path))
                self.thermo_kinetics_controller.set_xyzin(Path(path))
                self.link_controller.set_xyzin(Path(path))
                self.refresh()

        def refresh(self) -> None:
            self.workflow_list.clear()
            self.section_table.setRowCount(0)
            self.action_table.setRowCount(0)
            self._clear_contracts_table()
            self._clear_structure_tables()
            self._clear_gicforge_tables()
            self._clear_gf_tables()
            self._clear_qm_jobs_tables()
            self._clear_electronic_tables()
            self._clear_sefit_tables()
            self._clear_thermo_kinetics_tables()
            self._clear_link_tables()
            self._clear_workbench_tables()
            self._clear_help_tab()
            if self.controller.xyzin is None:
                self.path_label.setText("No MATRIX project loaded")
                self.gicforge_summary.setPlainText("No MATRIX project loaded")
                self.gf_summary.setPlainText("No MATRIX project loaded")
                self.qm_jobs_summary.setPlainText("No MATRIX project loaded")
                self.electronic_summary.setPlainText("No MATRIX project loaded")
                self.sefit_summary.setPlainText("No MATRIX project loaded")
                self.thermo_kinetics_summary.setPlainText("No MATRIX project loaded")
                self.link_summary.setPlainText("No MATRIX project loaded")
                self.contracts_summary.setPlainText("No MATRIX project loaded")
                self.help_summary.setPlainText(online_help_text())
                for summary in self.workbench_summaries.values():
                    summary.setPlainText("No MATRIX project loaded")
                self.details.setPlainText(
                    "\n".join(f"{spec.title}: {spec.description}" for spec in ORACLE_GUI_WINDOWS)
                )
                self.log_output.setPlainText("\n".join(self.controller.log_lines))
                self.run_button.setEnabled(False)
                return

            state = load_oracle_project_state(self.controller.xyzin)
            self.structure_controller.set_xyzin(state.xyzin)
            self.gicforge_controller.set_xyzin(state.xyzin)
            self.gf_controller.set_xyzin(state.xyzin)
            self.qm_jobs_controller.set_xyzin(state.xyzin)
            self.electronic_controller.set_xyzin(state.xyzin)
            self.sefit_controller.set_xyzin(state.xyzin)
            self.thermo_kinetics_controller.set_xyzin(state.xyzin)
            self.link_controller.set_xyzin(state.xyzin)
            self.structure_output_path.setText(str(state.xyzin))
            self._set_default_gicforge_outputs(state.xyzin)
            self._set_default_gf_outputs(state.xyzin)
            self._set_default_qm_jobs_outputs(state.xyzin)
            self._set_default_electronic_outputs(state.xyzin)
            self._set_default_sefit_outputs(state.xyzin)
            self._set_default_thermo_kinetics_outputs(state.xyzin)
            self._set_default_link_outputs(state.xyzin)
            self.path_label.setText(str(state.xyzin))
            for workflow in state.workflows:
                self.workflow_list.addItem(f"{workflow.status.value.upper():8s}  {workflow.title}")

            self.section_table.setRowCount(len(state.sections))
            for row, section in enumerate(state.sections):
                self.section_table.setItem(row, 0, QTableWidgetItem(section.name))
                self.section_table.setItem(row, 1, QTableWidgetItem(str(section.line_count)))
                self.section_table.setItem(row, 2, QTableWidgetItem(section.schema or ""))
            self.section_table.resizeColumnsToContents()
            self._populate_actions(state)
            self._populate_contracts_table(state.xyzin)
            self._populate_help_tab(state.xyzin)

            self.details.setPlainText(
                "\n".join(
                    [
                        f"atoms: {state.atom_count}",
                        f"point group: {state.point_group or 'unknown'}",
                        f"validation: {state.validation_status}",
                        "",
                        *state.validation_messages,
                    ]
                )
            )
            self.log_output.setPlainText("\n".join(self.controller.log_lines))
            self.run_button.setEnabled(
                self.process is None and any(action.enabled for action in self.current_actions)
            )
            self._populate_structure_tables(state.xyzin)
            self._populate_gicforge_tables(state.xyzin)
            self._populate_gf_tables(state.xyzin)
            self._populate_qm_jobs_tables(state.xyzin)
            self._populate_electronic_tables(state.xyzin)
            self._populate_sefit_tables(state.xyzin)
            self._populate_thermo_kinetics_tables(state.xyzin)
            self._populate_link_tables(state.xyzin)
            self._populate_workbench_tables(state.xyzin)

        def show_workflow_details(self, _item=None) -> None:
            if self.controller.xyzin is None:
                return
            state = load_oracle_project_state(self.controller.xyzin)
            row = self.workflow_list.currentRow()
            if row < 0 or row >= len(state.workflows):
                return
            workflow = state.workflows[row]
            QMessageBox.information(
                self,
                workflow.title,
                "\n".join(
                    [
                        f"Status: {workflow.status.value}",
                        workflow.message,
                        "",
                        "Required: " + (", ".join(workflow.required_sections) or "none"),
                        "Produced: " + (", ".join(workflow.produced_sections) or "none"),
                        "",
                        *self._spec_extra_lines(workflow.key),
                    ]
                ),
            )

        def _spec_extra_lines(self, key: str) -> list[str]:
            for spec in ORACLE_GUI_WINDOWS:
                if spec.key != key:
                    continue
                lines = [f"Category: {spec.category}"]
                if spec.capabilities:
                    lines.append("Capabilities:")
                    lines.extend(f"- {item}" for item in spec.capabilities)
                if spec.publication_exports:
                    lines.append("Publication export:")
                    lines.extend(f"- {item}" for item in spec.publication_exports)
                if spec.external_viewers:
                    lines.append("External viewers:")
                    lines.extend(f"- {item}" for item in spec.external_viewers)
                return lines
            return []

        def _populate_actions(self, state) -> None:
            self.current_actions = self.controller.actions(state)
            self.action_table.setRowCount(len(self.current_actions))
            for row, action in enumerate(self.current_actions):
                values = (
                    action.label,
                    "ready" if action.enabled else action.reason,
                    action.window_key,
                    action.shell_line,
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    if not action.enabled:
                        item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                    self.action_table.setItem(row, column, item)
            self.action_table.resizeColumnsToContents()

        def _build_contracts_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            self.contracts_summary = QTextEdit()
            self.contracts_summary.setReadOnly(True)
            self.contracts_summary.setMaximumHeight(90)
            layout.addWidget(self.contracts_summary)
            self.contracts_table = QTableWidget(0, 8)
            layout.addWidget(self.contracts_table, stretch=1)
            return tab

        def _populate_contracts_table(self, xyzin: Path) -> None:
            state = load_tool_contract_gui_state(xyzin)
            self.contracts_summary.setPlainText("\n".join(tool_contract_gui_state_lines(state)))
            self._fill_table(self.contracts_table, state.table)

        def _clear_contracts_table(self) -> None:
            summary = getattr(self, "contracts_summary", None)
            if summary is not None:
                summary.clear()
            table = getattr(self, "contracts_table", None)
            if table is not None:
                table.setRowCount(0)

        def _build_help_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            self.help_summary = QTextEdit()
            self.help_summary.setReadOnly(True)
            layout.addWidget(self.help_summary, stretch=1)
            return tab

        def _populate_help_tab(self, xyzin: Path) -> None:
            self.help_summary.setPlainText(online_help_text(xyzin=xyzin))

        def _clear_help_tab(self) -> None:
            summary = getattr(self, "help_summary", None)
            if summary is not None:
                summary.clear()

        def _build_workbench_tab(self, key: str) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            summary = QTextEdit()
            summary.setReadOnly(True)
            summary.setMaximumHeight(120)
            self.workbench_summaries[key] = summary
            layout.addWidget(summary)

            tables = QTabWidget()
            section_table = QTableWidget(0, 3)
            action_table = QTableWidget(0, 6)
            capability_table = QTableWidget(0, 1)
            export_table = QTableWidget(0, 2)
            self.workbench_section_tables[key] = section_table
            self.workbench_action_tables[key] = action_table
            self.workbench_capability_tables[key] = capability_table
            self.workbench_export_tables[key] = export_table
            tables.addTab(section_table, "Sections")
            tables.addTab(action_table, "Actions")
            tables.addTab(capability_table, "Capabilities")
            tables.addTab(export_table, "Exports")
            layout.addWidget(tables, stretch=1)
            return tab

        def _populate_workbench_tables(self, xyzin: Path) -> None:
            for key, _label in self.workbench_tabs:
                state = load_workbench_gui_state(xyzin, key)
                self.workbench_summaries[key].setPlainText(
                    "\n".join(workbench_gui_state_lines(state))
                )
                self._fill_table(self.workbench_section_tables[key], state.sections)
                self._fill_table(self.workbench_action_tables[key], state.actions)
                self._fill_table(self.workbench_capability_tables[key], state.capabilities)
                self._fill_table(self.workbench_export_tables[key], state.exports)

        def _clear_workbench_tables(self) -> None:
            for summary in getattr(self, "workbench_summaries", {}).values():
                summary.clear()
            for tables in (
                getattr(self, "workbench_section_tables", {}),
                getattr(self, "workbench_action_tables", {}),
                getattr(self, "workbench_capability_tables", {}),
                getattr(self, "workbench_export_tables", {}),
            ):
                for table in tables.values():
                    table.setRowCount(0)

        def run_selected_action(self, _item=None) -> None:
            row = self.action_table.currentRow()
            if row < 0 or row >= len(self.current_actions):
                QMessageBox.information(self, "ORACLE", "Select an action first.")
                return
            action = self.current_actions[row]
            if not action.enabled:
                QMessageBox.warning(self, action.label, action.reason)
                return
            if self.process is not None:
                QMessageBox.information(self, "ORACLE", "A command is already running.")
                return
            self._start_process(action)

        def _start_process(self, action: DashboardAction) -> None:
            self._start_command(action.command, action.label)

        def _start_command(self, command: OracleGuiCommand, label: str) -> None:
            if not self._ensure_command_sections(command, label):
                return
            prepared = self.controller.prepare_command(command)
            self.controller.log(f"$ {prepared.shell_line()}")
            self.log_output.setPlainText("\n".join(self.controller.log_lines))
            process = QProcess(self)
            if prepared.cwd is not None:
                process.setWorkingDirectory(str(prepared.cwd))
            process.setProgram(prepared.argv[0])
            process.setArguments(list(prepared.argv[1:]))
            process.readyReadStandardOutput.connect(self._read_process_stdout)
            process.readyReadStandardError.connect(self._read_process_stderr)
            process.errorOccurred.connect(lambda error: self._process_error(label, error))
            process.finished.connect(lambda code, _status: self._process_finished(label, code))
            self.process = process
            self.run_button.setEnabled(False)
            process.start()

        def _ensure_command_sections(self, command: OracleGuiCommand, title: str) -> bool:
            if self.controller.xyzin is None or not command.required_sections:
                return True
            state = load_oracle_project_state(self.controller.xyzin)
            present = set(state.section_names)
            missing = tuple(
                section for section in command.required_sections if section not in present
            )
            if not missing:
                return True
            QMessageBox.warning(self, title, missing_sections_message(missing))
            return False

        def _read_process_stdout(self) -> None:
            if self.process is None:
                return
            text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
            self.controller.log(text.rstrip())
            self.log_output.setPlainText("\n".join(self.controller.log_lines))

        def _read_process_stderr(self) -> None:
            if self.process is None:
                return
            text = bytes(self.process.readAllStandardError()).decode(errors="replace")
            self.controller.log(text.rstrip())
            self.log_output.setPlainText("\n".join(self.controller.log_lines))

        def _process_finished(self, label: str, code: int) -> None:
            self.controller.log(f"[exit {int(code)}] {label}")
            self.process = None
            if self.pending_xyzin_after_run is not None and self.pending_xyzin_after_run.exists():
                self.controller.set_xyzin(self.pending_xyzin_after_run)
                self.structure_controller.set_xyzin(self.pending_xyzin_after_run)
                self.gicforge_controller.set_xyzin(self.pending_xyzin_after_run)
                self.gf_controller.set_xyzin(self.pending_xyzin_after_run)
                self.qm_jobs_controller.set_xyzin(self.pending_xyzin_after_run)
                self.electronic_controller.set_xyzin(self.pending_xyzin_after_run)
                self.sefit_controller.set_xyzin(self.pending_xyzin_after_run)
                self.thermo_kinetics_controller.set_xyzin(self.pending_xyzin_after_run)
                self.link_controller.set_xyzin(self.pending_xyzin_after_run)
            self.pending_xyzin_after_run = None
            self.refresh()

        def _process_error(self, label: str, error) -> None:
            self.controller.log(f"[process error {int(error)}] {label}")
            if self.process is not None and self.process.state() == QProcess.NotRunning:
                self.process = None
                self.refresh()

        def _build_structure_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            source_row = QHBoxLayout()
            source_row.addWidget(QLabel("Source"))
            self.structure_source_path = QLineEdit()
            browse_source = QPushButton("Browse")
            browse_source.clicked.connect(self.browse_structure_source)
            source_row.addWidget(self.structure_source_path, stretch=1)
            source_row.addWidget(browse_source)
            layout.addLayout(source_row)

            output_row = QHBoxLayout()
            output_row.addWidget(QLabel("Output xyzin"))
            self.structure_output_path = QLineEdit()
            browse_output = QPushButton("Save As")
            browse_output.clicked.connect(self.browse_structure_output)
            output_row.addWidget(self.structure_output_path, stretch=1)
            output_row.addWidget(browse_output)
            layout.addLayout(output_row)

            controls = QHBoxLayout()
            self.structure_source_kind = QComboBox()
            self.structure_source_kind.addItems(
                ("auto", "xyz", "enriched_xyz", "gaussian", "molpro", "mrcc")
            )
            preprocess_button = QPushButton("Preprocess")
            preprocess_button.clicked.connect(self.run_structure_preprocess)
            avogadro_button = QPushButton("Avogadro")
            avogadro_button.clicked.connect(self.run_structure_avogadro)
            fragments_button = QPushButton("Build Fragments")
            fragments_button.clicked.connect(self.run_structure_fragments)
            controls.addWidget(QLabel("Kind"))
            controls.addWidget(self.structure_source_kind)
            controls.addWidget(preprocess_button)
            controls.addWidget(avogadro_button)
            controls.addWidget(fragments_button)
            controls.addStretch(1)
            layout.addLayout(controls)

            self.structure_table_tabs = QTabWidget()
            self.structure_bond_table = QTableWidget(0, 2)
            self.structure_ring_table = QTableWidget(0, 3)
            self.structure_synthon_table = QTableWidget(0, 8)
            self.structure_fragment_table = QTableWidget(0, 5)
            self.structure_table_tabs.addTab(self.structure_bond_table, "Bonds")
            self.structure_table_tabs.addTab(self.structure_ring_table, "Rings")
            self.structure_table_tabs.addTab(self.structure_synthon_table, "Synthons")
            self.structure_table_tabs.addTab(self.structure_fragment_table, "Fragments")
            layout.addWidget(self.structure_table_tabs, stretch=1)
            return tab

        def browse_structure_source(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Import molecule source",
                str(Path.cwd()),
                "Molecule sources (*.xyz *.xyzin *.gjf *.gau *.com *.inp *.log *.out *.zmat *.zmt);;All files (*)",
            )
            if not path:
                return
            source = Path(path)
            self.structure_source_path.setText(str(source))
            if not self.structure_output_path.text().strip():
                self.structure_output_path.setText(str(default_preprocess_output(source)))

        def browse_structure_output(self) -> None:
            path, _selected = QFileDialog.getSaveFileName(
                self,
                "Write MATRIX xyzin",
                self.structure_output_path.text().strip() or str(Path.cwd() / "molecule.xyzin"),
                "MATRIX xyzin (*.xyzin *.xyz);;All files (*)",
            )
            if path:
                self.structure_output_path.setText(path)

        def run_structure_preprocess(self) -> None:
            source_text = self.structure_source_path.text().strip()
            output_text = self.structure_output_path.text().strip()
            if not source_text or not output_text:
                QMessageBox.warning(
                    self, "ORACLE Import", "Select source and output files first."
                )
                return
            if self.process is not None:
                QMessageBox.information(self, "ORACLE", "A command is already running.")
                return
            output = Path(output_text)
            command = self.structure_controller.preprocess_command(
                Path(source_text),
                output,
                source_kind=self.structure_source_kind.currentText(),
            )
            self.pending_xyzin_after_run = output
            self._start_command(command, command.label)

        def run_structure_avogadro(self) -> None:
            if self.controller.xyzin is None:
                QMessageBox.warning(self, "Avogadro", "Open or preprocess a MATRIX xyzin first.")
                return
            if self.process is not None:
                QMessageBox.information(self, "ORACLE", "A command is already running.")
                return
            self.structure_controller.set_xyzin(self.controller.xyzin)
            command = self.structure_controller.avogadro_command()
            self._start_command(command, command.label)

        def run_structure_fragments(self) -> None:
            if self.controller.xyzin is None:
                QMessageBox.warning(self, "Fragments", "Open or preprocess a MATRIX xyzin first.")
                return
            if self.process is not None:
                QMessageBox.information(self, "ORACLE", "A command is already running.")
                return
            self.structure_controller.set_xyzin(self.controller.xyzin)
            command = self.structure_controller.fragments_command("build")
            self._start_command(command, command.label)

        def _populate_structure_tables(self, xyzin: Path) -> None:
            state = load_structure_gui_state(xyzin)
            self._fill_table(self.structure_bond_table, state.topology_bonds)
            self._fill_table(self.structure_ring_table, state.topology_rings)
            self._fill_table(self.structure_synthon_table, state.synthons)
            self._fill_table(self.structure_fragment_table, state.fragments)

        def _clear_structure_tables(self) -> None:
            for table in (
                getattr(self, "structure_bond_table", None),
                getattr(self, "structure_ring_table", None),
                getattr(self, "structure_synthon_table", None),
                getattr(self, "structure_fragment_table", None),
            ):
                if table is not None:
                    table.setRowCount(0)

        def _build_gicforge_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            option_row = QHBoxLayout()
            self.gicforge_symmetrize = QCheckBox("Symmetrize")
            self.gicforge_symmetrize.setChecked(True)
            self.gicforge_sycart = QCheckBox("SYCART")
            self.gicforge_sycart.setChecked(True)
            self.gicforge_improper = QCheckBox("Improper dihedrals")
            self.gicforge_improper.setChecked(True)
            build_button = QPushButton("Build GICs")
            build_button.clicked.connect(self.run_gicforge_build)
            option_row.addWidget(self.gicforge_symmetrize)
            option_row.addWidget(self.gicforge_sycart)
            option_row.addWidget(self.gicforge_improper)
            option_row.addWidget(build_button)
            option_row.addStretch(1)
            layout.addLayout(option_row)

            report_row = QHBoxLayout()
            report_row.addWidget(QLabel("Report"))
            self.gicforge_report_output = QLineEdit()
            report_browse = QPushButton("Save As")
            report_browse.clicked.connect(self.browse_gicforge_report_output)
            report_button = QPushButton("Write Report")
            report_button.clicked.connect(self.run_gicforge_report)
            report_row.addWidget(self.gicforge_report_output, stretch=1)
            report_row.addWidget(report_browse)
            report_row.addWidget(report_button)
            layout.addLayout(report_row)

            bmatrix_row = QHBoxLayout()
            bmatrix_row.addWidget(QLabel("B Matrix"))
            self.gicforge_bmatrix_output = QLineEdit()
            bmatrix_browse = QPushButton("Save As")
            bmatrix_browse.clicked.connect(self.browse_gicforge_bmatrix_output)
            bmatrix_button = QPushButton("Evaluate B")
            bmatrix_button.clicked.connect(self.run_gicforge_bmatrix)
            bmatrix_row.addWidget(self.gicforge_bmatrix_output, stretch=1)
            bmatrix_row.addWidget(bmatrix_browse)
            bmatrix_row.addWidget(bmatrix_button)
            layout.addLayout(bmatrix_row)

            gaussian_row = QHBoxLayout()
            gaussian_row.addWidget(QLabel("Gaussian"))
            self.gicforge_gaussian_output = QLineEdit()
            gaussian_browse = QPushButton("Save As")
            gaussian_browse.clicked.connect(self.browse_gicforge_gaussian_output)
            self.gicforge_gaussian_route = QLineEdit("#p hf/sto-3g opt=readallgic")
            gaussian_button = QPushButton("Write Input")
            gaussian_button.clicked.connect(self.run_gicforge_gaussian_input)
            gaussian_row.addWidget(self.gicforge_gaussian_output, stretch=1)
            gaussian_row.addWidget(gaussian_browse)
            gaussian_row.addWidget(QLabel("Route"))
            gaussian_row.addWidget(self.gicforge_gaussian_route)
            gaussian_row.addWidget(gaussian_button)
            layout.addLayout(gaussian_row)

            self.gicforge_summary = QTextEdit()
            self.gicforge_summary.setReadOnly(True)
            self.gicforge_summary.setMaximumHeight(150)
            layout.addWidget(self.gicforge_summary)

            self.gicforge_table_tabs = QTabWidget()
            self.gicforge_primitive_table = QTableWidget(0, 8)
            self.gicforge_frozen_table = QTableWidget(0, 7)
            self.gicforge_symmetry_table = QTableWidget(0, 5)
            self.gicforge_diagnostics_table = QTableWidget(0, 2)
            self.gicforge_table_tabs.addTab(self.gicforge_primitive_table, "Primitives")
            self.gicforge_table_tabs.addTab(self.gicforge_frozen_table, "Frozen GICs")
            self.gicforge_table_tabs.addTab(self.gicforge_symmetry_table, "Symmetry")
            self.gicforge_table_tabs.addTab(self.gicforge_diagnostics_table, "Diagnostics")
            layout.addWidget(self.gicforge_table_tabs, stretch=1)
            return tab

        def browse_gicforge_report_output(self) -> None:
            self._browse_gicforge_output(
                self.gicforge_report_output,
                "Write GICForge report",
                "Text reports (*.txt);;All files (*)",
            )

        def browse_gicforge_bmatrix_output(self) -> None:
            self._browse_gicforge_output(
                self.gicforge_bmatrix_output,
                "Write GIC B matrix",
                "Text files (*.txt);;All files (*)",
            )

        def browse_gicforge_gaussian_output(self) -> None:
            self._browse_gicforge_output(
                self.gicforge_gaussian_output,
                "Write Gaussian GIC input",
                "Gaussian input (*.gjf *.com);;All files (*)",
            )

        def _browse_gicforge_output(
            self,
            field: QLineEdit,
            title: str,
            file_filter: str,
        ) -> None:
            path, _selected = QFileDialog.getSaveFileName(
                self,
                title,
                field.text().strip() or str(Path.cwd()),
                file_filter,
            )
            if path:
                field.setText(path)

        def run_gicforge_build(self) -> None:
            if not self._ensure_gicforge_ready_for_command("GICForge"):
                return
            self.gicforge_controller.set_xyzin(self.controller.xyzin)
            command = self.gicforge_controller.build_command(
                symmetrize=self.gicforge_symmetrize.isChecked(),
                sycart=self.gicforge_sycart.isChecked(),
                improper_dihedrals=self.gicforge_improper.isChecked(),
            )
            self._start_command(command, command.label)

        def run_gicforge_report(self) -> None:
            if not self._ensure_gicforge_ready_for_command("GICForge report"):
                return
            if not self._ensure_gicforge_section_ready("GICForge report"):
                return
            output = self._gicforge_output_path(
                self.gicforge_report_output,
                default_gicforge_report_output,
            )
            command = self.gicforge_controller.report_command(output)
            self._start_command(command, command.label)

        def run_gicforge_bmatrix(self) -> None:
            if not self._ensure_gicforge_ready_for_command("GICForge B matrix"):
                return
            if not self._ensure_gicforge_section_ready("GICForge B matrix"):
                return
            output = self._gicforge_output_path(
                self.gicforge_bmatrix_output,
                default_gicforge_bmatrix_output,
            )
            command = self.gicforge_controller.bmatrix_command(output)
            self._start_command(command, command.label)

        def run_gicforge_gaussian_input(self) -> None:
            if not self._ensure_gicforge_ready_for_command("Gaussian GIC input"):
                return
            if not self._ensure_gicforge_section_ready("Gaussian GIC input"):
                return
            output = self._gicforge_output_path(
                self.gicforge_gaussian_output,
                default_gicforge_gaussian_output,
            )
            route = self.gicforge_gaussian_route.text().strip() or "#p hf/sto-3g opt=readallgic"
            command = self.gicforge_controller.gaussian_input_command(output, route=route)
            self._start_command(command, command.label)

        def _ensure_gicforge_ready_for_command(self, title: str) -> bool:
            if self.controller.xyzin is None:
                QMessageBox.warning(self, title, "Open or preprocess a MATRIX xyzin first.")
                return False
            if self.process is not None:
                QMessageBox.information(self, "ORACLE", "A command is already running.")
                return False
            self.gicforge_controller.set_xyzin(self.controller.xyzin)
            return True

        def _ensure_gicforge_section_ready(self, title: str) -> bool:
            if self.controller.xyzin is None:
                return False
            state = load_gicforge_gui_state(self.controller.xyzin)
            if state.ready:
                return True
            message = "\n".join(state.messages) or "Build GICs first."
            QMessageBox.warning(self, title, message)
            return False

        def _gicforge_output_path(self, field: QLineEdit, default_factory) -> Path:
            if self.controller.xyzin is None:
                raise ValueError("no MATRIX xyzin project is loaded")
            text = field.text().strip()
            output = Path(text) if text else default_factory(self.controller.xyzin)
            field.setText(str(output))
            return output

        def _set_default_gicforge_outputs(self, xyzin: Path) -> None:
            self.gicforge_report_output.setText(str(default_gicforge_report_output(xyzin)))
            self.gicforge_bmatrix_output.setText(str(default_gicforge_bmatrix_output(xyzin)))
            self.gicforge_gaussian_output.setText(str(default_gicforge_gaussian_output(xyzin)))

        def _populate_gicforge_tables(self, xyzin: Path) -> None:
            state = load_gicforge_gui_state(xyzin)
            self.gicforge_summary.setPlainText("\n".join(gicforge_gui_state_lines(state)))
            self._fill_table(self.gicforge_primitive_table, state.primitives)
            self._fill_table(self.gicforge_frozen_table, state.frozen_gics)
            self._fill_table(self.gicforge_symmetry_table, state.symmetry_groups)
            self._fill_table(self.gicforge_diagnostics_table, state.diagnostics)

        def _clear_gicforge_tables(self) -> None:
            summary = getattr(self, "gicforge_summary", None)
            if summary is not None:
                summary.clear()
            for table in (
                getattr(self, "gicforge_primitive_table", None),
                getattr(self, "gicforge_frozen_table", None),
                getattr(self, "gicforge_symmetry_table", None),
                getattr(self, "gicforge_diagnostics_table", None),
            ):
                if table is not None:
                    table.setRowCount(0)

        def _build_gf_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            source_row = QHBoxLayout()
            source_row.addWidget(QLabel("FCHK"))
            self.gf_fchk_path = QLineEdit()
            fchk_browse = QPushButton("Browse")
            fchk_browse.clicked.connect(self.browse_gf_fchk)
            source_row.addWidget(self.gf_fchk_path, stretch=1)
            source_row.addWidget(fchk_browse)
            layout.addLayout(source_row)

            output_row = QHBoxLayout()
            output_row.addWidget(QLabel("Report"))
            self.gf_report_output = QLineEdit()
            report_browse = QPushButton("Save As")
            report_browse.clicked.connect(self.browse_gf_report_output)
            output_row.addWidget(self.gf_report_output, stretch=1)
            output_row.addWidget(report_browse)
            output_row.addWidget(QLabel("CSV dir"))
            self.gf_csv_dir = QLineEdit()
            csv_browse = QPushButton("Browse")
            csv_browse.clicked.connect(self.browse_gf_csv_dir)
            output_row.addWidget(self.gf_csv_dir, stretch=1)
            output_row.addWidget(csv_browse)
            layout.addLayout(output_row)

            scale_row = QHBoxLayout()
            scale_row.addWidget(QLabel("Scale file"))
            self.gf_scale_file = QLineEdit()
            scale_browse = QPushButton("Browse")
            scale_browse.clicked.connect(self.browse_gf_scale_file)
            self.gf_scale_records = QLineEdit()
            scale_row.addWidget(self.gf_scale_file, stretch=1)
            scale_row.addWidget(scale_browse)
            scale_row.addWidget(QLabel("Scale"))
            scale_row.addWidget(self.gf_scale_records, stretch=1)
            layout.addLayout(scale_row)

            scale_class_row = QHBoxLayout()
            scale_class_row.addWidget(QLabel("Scale classes"))
            self.gf_scale_class_records = QLineEdit()
            self.gf_scale_class_records.setPlaceholderText("CH:0.970:R(1,6)|R(2,7); class ...")
            preview_scaling_button = QPushButton("Preview Scaling")
            preview_scaling_button.clicked.connect(self.preview_gf_scaling)
            scale_class_row.addWidget(self.gf_scale_class_records, stretch=1)
            scale_class_row.addWidget(preview_scaling_button)
            layout.addLayout(scale_class_row)

            options_row = QHBoxLayout()
            self.gf_symmetry_blocks = QCheckBox("Symmetry blocks")
            self.gf_symmetry_blocks.setChecked(True)
            self.gf_local = QCheckBox("Local")
            self.gf_subtract_electrostatic = QCheckBox("Subtract electrostatic")
            self.gf_subtract_uff_vdw = QCheckBox("Subtract UFF vdW")
            self.gf_write_section = QCheckBox("Update #GF_PED")
            self.gf_write_section.setChecked(True)
            self.gf_force_threshold = QLineEdit()
            self.gf_force_threshold.setPlaceholderText("force threshold")
            self.gf_nonbonded_14_scale = QLineEdit("0.5")
            run_button = QPushButton("Run TRINITY harmonic")
            run_button.clicked.connect(self.run_gf)
            options_row.addWidget(self.gf_symmetry_blocks)
            options_row.addWidget(self.gf_local)
            options_row.addWidget(self.gf_subtract_electrostatic)
            options_row.addWidget(self.gf_subtract_uff_vdw)
            options_row.addWidget(self.gf_write_section)
            options_row.addWidget(QLabel("Threshold"))
            options_row.addWidget(self.gf_force_threshold)
            options_row.addWidget(QLabel("1-4"))
            options_row.addWidget(self.gf_nonbonded_14_scale)
            options_row.addWidget(run_button)
            layout.addLayout(options_row)

            self.gf_summary = QTextEdit()
            self.gf_summary.setReadOnly(True)
            self.gf_summary.setMaximumHeight(150)
            layout.addWidget(self.gf_summary)

            self.gf_table_tabs = QTabWidget()
            self.gf_frequency_table = QTableWidget(0, 3)
            self.gf_gic_table = QTableWidget(0, 7)
            self.gf_ped_table = QTableWidget(0, 3)
            self.gf_diagnostics_table = QTableWidget(0, 2)
            self.gf_table_tabs.addTab(self.gf_frequency_table, "Frequencies")
            self.gf_table_tabs.addTab(self.gf_gic_table, "GICs")
            self.gf_table_tabs.addTab(self.gf_ped_table, "PED")
            self.gf_table_tabs.addTab(self.gf_diagnostics_table, "Diagnostics")
            layout.addWidget(self.gf_table_tabs, stretch=1)
            return tab

        def browse_gf_fchk(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select Gaussian FCHK",
                str(Path.cwd()),
                "Gaussian FCHK (*.fchk *.fch);;All files (*)",
            )
            if path:
                self.gf_fchk_path.setText(path)

        def browse_gf_report_output(self) -> None:
            path, _selected = QFileDialog.getSaveFileName(
                self,
                "Write TRINITY report",
                self.gf_report_output.text().strip() or str(Path.cwd() / "gf_ped_report.txt"),
                "Text reports (*.txt *.report);;All files (*)",
            )
            if path:
                self.gf_report_output.setText(path)

        def browse_gf_csv_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select TRINITY CSV directory",
                self.gf_csv_dir.text().strip() or str(Path.cwd()),
            )
            if path:
                self.gf_csv_dir.setText(path)

        def browse_gf_scale_file(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select GF scaling file",
                str(Path.cwd()),
                "Scaling files (*.txt *.csv *.dat);;All files (*)",
            )
            if path:
                self.gf_scale_file.setText(path)

        def run_gf(self) -> None:
            if not self._ensure_gf_ready_for_command("TRINITY"):
                return
            if not self._ensure_gf_inputs_ready("TRINITY"):
                return
            force_threshold = self._optional_float_field(
                self.gf_force_threshold,
                "TRINITY",
                "force threshold",
            )
            if force_threshold is False:
                return
            nonbonded_14_scale = self._optional_float_field(
                self.gf_nonbonded_14_scale,
                "TRINITY",
                "1-4 scale",
                default=0.5,
            )
            if nonbonded_14_scale is False:
                return
            self.gf_controller.set_xyzin(self.controller.xyzin)
            command = self.gf_controller.run_command(
                fchk=self._optional_path_text(self.gf_fchk_path),
                out=self._gf_output_path(self.gf_report_output, default_gf_report_output),
                csv_dir=self._gf_output_path(self.gf_csv_dir, default_gf_csv_dir),
                scale_file=self._optional_path_text(self.gf_scale_file),
                scale_records=self._scale_records(),
                scale_class_records=self._scale_class_records(),
                local=self.gf_local.isChecked(),
                symmetry_blocks=self.gf_symmetry_blocks.isChecked(),
                force_threshold=force_threshold,
                subtract_electrostatic=self.gf_subtract_electrostatic.isChecked(),
                subtract_uff_vdw=self.gf_subtract_uff_vdw.isChecked(),
                nonbonded_14_scale=nonbonded_14_scale,
                write_section=self.gf_write_section.isChecked(),
            )
            self._start_command(command, command.label)

        def preview_gf_scaling(self) -> None:
            if not self._ensure_gf_ready_for_command("GF scaling preview"):
                return
            if not self._ensure_gf_gic_ready("GF scaling preview"):
                return
            self.gf_controller.set_xyzin(self.controller.xyzin)
            command = self.gf_controller.scaling_preview_command(
                scale_file=self._optional_path_text(self.gf_scale_file),
                scale_records=self._scale_records(),
                scale_class_records=self._scale_class_records(),
            )
            self._start_command(command, command.label)

        def _ensure_gf_ready_for_command(self, title: str) -> bool:
            if self.controller.xyzin is None:
                QMessageBox.warning(self, title, "Open or preprocess a MATRIX xyzin first.")
                return False
            if self.process is not None:
                QMessageBox.information(self, "ORACLE", "A command is already running.")
                return False
            self.gf_controller.set_xyzin(self.controller.xyzin)
            return True

        def _ensure_gf_gic_ready(self, title: str) -> bool:
            if self.controller.xyzin is None:
                return False
            state = load_oracle_project_state(self.controller.xyzin)
            if "GIC" in set(state.section_names):
                return True
            QMessageBox.warning(self, title, missing_sections_message(("GIC",)))
            return False

        def _ensure_gf_inputs_ready(self, title: str) -> bool:
            if self.controller.xyzin is None:
                return False
            state = load_oracle_project_state(self.controller.xyzin)
            sections = set(state.section_names)
            missing = []
            if "GIC" not in sections:
                missing.append("GIC")
            if not self.gf_fchk_path.text().strip() and "CARTESIAN_HESSIAN" not in sections:
                missing.append("CARTESIAN_HESSIAN")
            if not missing:
                return True
            message = missing_sections_message(tuple(missing))
            if "CARTESIAN_HESSIAN" in missing:
                message += "\n\nAlternative: select a Gaussian FCHK file in the TRINITY tab."
            QMessageBox.warning(self, title, message)
            return False

        def _optional_float_field(
            self,
            field: QLineEdit,
            title: str,
            label: str,
            *,
            default=None,
        ):
            text = field.text().strip()
            if not text:
                return default
            try:
                return float(text)
            except ValueError:
                QMessageBox.warning(self, title, f"Invalid {label}: {text}")
                return False

        def _optional_path_text(self, field: QLineEdit) -> Path | None:
            text = field.text().strip()
            return Path(text) if text else None

        def _scale_records(self) -> tuple[str, ...]:
            text = self.gf_scale_records.text().strip()
            if not text:
                return ()
            return tuple(item.strip() for item in text.split(";") if item.strip())

        def _scale_class_records(self) -> tuple[str, ...]:
            text = self.gf_scale_class_records.text().strip()
            if not text:
                return ()
            return tuple(item.strip() for item in text.split(";") if item.strip())

        def _gf_output_path(self, field: QLineEdit, default_factory) -> Path:
            if self.controller.xyzin is None:
                raise ValueError("no MATRIX xyzin project is loaded")
            text = field.text().strip()
            output = Path(text) if text else default_factory(self.controller.xyzin)
            field.setText(str(output))
            return output

        def _set_default_gf_outputs(self, xyzin: Path) -> None:
            self.gf_report_output.setText(str(default_gf_report_output(xyzin)))
            self.gf_csv_dir.setText(str(default_gf_csv_dir(xyzin)))

        def _populate_gf_tables(self, xyzin: Path) -> None:
            state = load_gf_gui_state(xyzin)
            self.gf_summary.setPlainText("\n".join(gf_gui_state_lines(state)))
            self._fill_table(self.gf_frequency_table, state.frequencies)
            self._fill_table(self.gf_gic_table, state.gics)
            self._fill_table(self.gf_ped_table, state.ped)
            self._fill_table(self.gf_diagnostics_table, state.diagnostics)

        def _clear_gf_tables(self) -> None:
            summary = getattr(self, "gf_summary", None)
            if summary is not None:
                summary.clear()
            for table in (
                getattr(self, "gf_frequency_table", None),
                getattr(self, "gf_gic_table", None),
                getattr(self, "gf_ped_table", None),
                getattr(self, "gf_diagnostics_table", None),
            ):
                if table is not None:
                    table.setRowCount(0)

        def _build_qm_jobs_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            gaussian_input_row = QHBoxLayout()
            gaussian_input_row.addWidget(QLabel("Gaussian input"))
            self.qm_gaussian_input_output = QLineEdit()
            gaussian_input_browse = QPushButton("Save As")
            gaussian_input_browse.clicked.connect(self.browse_qm_gaussian_input_output)
            self.qm_gaussian_route = QLineEdit("#p hf/sto-3g opt=readallgic")
            gaussian_input_button = QPushButton("Write Input")
            gaussian_input_button.clicked.connect(self.run_qm_gaussian_input)
            gaussian_input_row.addWidget(self.qm_gaussian_input_output, stretch=1)
            gaussian_input_row.addWidget(gaussian_input_browse)
            gaussian_input_row.addWidget(QLabel("Route"))
            gaussian_input_row.addWidget(self.qm_gaussian_route, stretch=1)
            gaussian_input_row.addWidget(gaussian_input_button)
            layout.addLayout(gaussian_input_row)

            job_row = QHBoxLayout()
            job_row.addWidget(QLabel("Job dir"))
            self.qm_gaussian_workdir = QLineEdit()
            workdir_browse = QPushButton("Browse")
            workdir_browse.clicked.connect(self.browse_qm_gaussian_workdir)
            self.qm_gaussian_job_input = QLineEdit()
            job_input_browse = QPushButton("Input")
            job_input_browse.clicked.connect(self.browse_qm_gaussian_job_input)
            self.qm_gaussian_executable = QLineEdit()
            self.qm_gaussian_executable.setPlaceholderText("matrix-gdv (preferred)")
            self.qm_gaussian_background = QCheckBox("Background")
            self.qm_gaussian_timeout = QLineEdit()
            self.qm_gaussian_timeout.setPlaceholderText("timeout")
            status_button = QPushButton("Status")
            status_button.clicked.connect(self.run_qm_gaussian_status)
            run_button = QPushButton("Run")
            run_button.clicked.connect(self.run_qm_gaussian_run)
            job_row.addWidget(self.qm_gaussian_workdir, stretch=1)
            job_row.addWidget(workdir_browse)
            job_row.addWidget(QLabel("Input"))
            job_row.addWidget(self.qm_gaussian_job_input, stretch=1)
            job_row.addWidget(job_input_browse)
            job_row.addWidget(QLabel("Exe"))
            job_row.addWidget(self.qm_gaussian_executable)
            job_row.addWidget(self.qm_gaussian_background)
            job_row.addWidget(QLabel("Timeout"))
            job_row.addWidget(self.qm_gaussian_timeout)
            job_row.addWidget(status_button)
            job_row.addWidget(run_button)
            layout.addLayout(job_row)

            remote_row = QHBoxLayout()
            remote_row.addWidget(QLabel("Remote"))
            self.qm_remote_input = QLineEdit()
            remote_input_browse = QPushButton("Input")
            remote_input_browse.clicked.connect(self.browse_qm_remote_input)
            self.qm_remote_engine = QComboBox()
            self.qm_remote_engine.addItems(("gdv32", "g16", "molpro", "orca"))
            self.qm_remote_host = QComboBox()
            try:
                remote_names = tuple(
                    machine.name
                    for machine in load_environment(missing_ok=True).remote_machines
                    if machine.enabled
                )
            except (OSError, ValueError):
                remote_names = ()
            self.qm_remote_host.addItems(remote_names)
            self.qm_remote_job = QLineEdit()
            self.qm_remote_job.setPlaceholderText("job id")
            self.qm_remote_destination = QLineEdit("remote_qm_runs")
            remote_dest_browse = QPushButton("Dest")
            remote_dest_browse.clicked.connect(self.browse_qm_remote_destination)
            self.qm_remote_promote = QComboBox()
            self.qm_remote_promote.addItems(
                (
                    "none",
                    "auto",
                    "molpro",
                    "gaussian-log-hessian",
                    "gaussian-rovib",
                    "gaussian-electronic",
                    "gaussian-fchk",
                    "orca",
                )
            )
            remote_submit_button = QPushButton("Submit")
            remote_submit_button.clicked.connect(self.run_qm_remote_submit)
            remote_status_button = QPushButton("Status")
            remote_status_button.clicked.connect(self.run_qm_remote_status)
            remote_fetch_button = QPushButton("Fetch")
            remote_fetch_button.clicked.connect(self.run_qm_remote_fetch)
            remote_row.addWidget(self.qm_remote_engine)
            remote_row.addWidget(self.qm_remote_input, stretch=1)
            remote_row.addWidget(remote_input_browse)
            remote_row.addWidget(QLabel("Host"))
            remote_row.addWidget(self.qm_remote_host)
            remote_row.addWidget(QLabel("Job"))
            remote_row.addWidget(self.qm_remote_job)
            remote_row.addWidget(QLabel("Dest"))
            remote_row.addWidget(self.qm_remote_destination)
            remote_row.addWidget(remote_dest_browse)
            remote_row.addWidget(QLabel("Promote"))
            remote_row.addWidget(self.qm_remote_promote)
            remote_row.addWidget(remote_submit_button)
            remote_row.addWidget(remote_status_button)
            remote_row.addWidget(remote_fetch_button)
            layout.addLayout(remote_row)

            formchk_row = QHBoxLayout()
            formchk_row.addWidget(QLabel("CHK"))
            self.qm_chk_path = QLineEdit()
            chk_browse = QPushButton("Browse")
            chk_browse.clicked.connect(self.browse_qm_chk_path)
            self.qm_fchk_output = QLineEdit()
            fchk_output_browse = QPushButton("Save As")
            fchk_output_browse.clicked.connect(self.browse_qm_fchk_output)
            self.qm_formchk_executable = QLineEdit()
            self.qm_formchk_executable.setPlaceholderText("formchk")
            self.qm_formchk_timeout = QLineEdit()
            self.qm_formchk_timeout.setPlaceholderText("timeout")
            formchk_button = QPushButton("Formchk")
            formchk_button.clicked.connect(self.run_qm_gaussian_formchk)
            formchk_row.addWidget(self.qm_chk_path, stretch=1)
            formchk_row.addWidget(chk_browse)
            formchk_row.addWidget(QLabel("FCHK out"))
            formchk_row.addWidget(self.qm_fchk_output, stretch=1)
            formchk_row.addWidget(fchk_output_browse)
            formchk_row.addWidget(QLabel("Exe"))
            formchk_row.addWidget(self.qm_formchk_executable)
            formchk_row.addWidget(QLabel("Timeout"))
            formchk_row.addWidget(self.qm_formchk_timeout)
            formchk_row.addWidget(formchk_button)
            layout.addLayout(formchk_row)

            fchk_row = QHBoxLayout()
            fchk_row.addWidget(QLabel("FCHK"))
            self.qm_fchk_path = QLineEdit()
            fchk_browse = QPushButton("Browse")
            fchk_browse.clicked.connect(self.browse_qm_fchk_path)
            self.qm_write_cartesian_hessian = QCheckBox("#CARTESIAN_HESSIAN")
            self.qm_write_cartesian_hessian.setChecked(True)
            self.qm_write_normal_modes = QCheckBox("#NORMAL_MODES")
            self.qm_write_normal_modes.setChecked(True)
            self.qm_write_qff = QCheckBox("#QFF")
            self.qm_write_qff.setChecked(True)
            self.qm_write_fchk_electronic = QCheckBox("#ELECTRONIC")
            self.qm_write_fchk_electronic.setChecked(True)
            self.qm_write_fchk_orbitals = QCheckBox("#ORBITALS")
            self.qm_write_fchk_orbitals.setChecked(True)
            fchk_summary_button = QPushButton("Summary")
            fchk_summary_button.clicked.connect(self.run_qm_fchk_summary)
            promote_fchk_button = QPushButton("Promote FCHK")
            promote_fchk_button.clicked.connect(self.run_qm_promote_fchk)
            fchk_row.addWidget(self.qm_fchk_path, stretch=1)
            fchk_row.addWidget(fchk_browse)
            fchk_row.addWidget(self.qm_write_cartesian_hessian)
            fchk_row.addWidget(self.qm_write_normal_modes)
            fchk_row.addWidget(self.qm_write_qff)
            fchk_row.addWidget(self.qm_write_fchk_electronic)
            fchk_row.addWidget(self.qm_write_fchk_orbitals)
            fchk_row.addWidget(fchk_summary_button)
            fchk_row.addWidget(promote_fchk_button)
            layout.addLayout(fchk_row)

            rovib_row = QHBoxLayout()
            rovib_row.addWidget(QLabel("Gaussian log"))
            self.qm_log_path = QLineEdit()
            log_browse = QPushButton("Browse")
            log_browse.clicked.connect(self.browse_qm_log_path)
            self.qm_write_vibrational = QCheckBox("#VIBRATIONAL")
            self.qm_write_vibrational.setChecked(True)
            self.qm_write_rotational = QCheckBox("#ROTATIONAL")
            self.qm_write_rotational.setChecked(True)
            self.qm_write_deltabvib = QCheckBox("#DELTABVIB")
            self.qm_write_deltabvib.setChecked(True)
            self.qm_invert_imaginary = QCheckBox("Invert imaginary")
            self.qm_invert_imaginary.setChecked(True)
            self.qm_exclude_modes = QLineEdit()
            self.qm_exclude_modes.setPlaceholderText("exclude modes")
            log_summary_button = QPushButton("Summary")
            log_summary_button.clicked.connect(self.run_qm_log_summary)
            promote_rovib_button = QPushButton("Promote Rovib")
            promote_rovib_button.clicked.connect(self.run_qm_promote_rovib)
            rovib_row.addWidget(self.qm_log_path, stretch=1)
            rovib_row.addWidget(log_browse)
            rovib_row.addWidget(self.qm_write_vibrational)
            rovib_row.addWidget(self.qm_write_rotational)
            rovib_row.addWidget(self.qm_write_deltabvib)
            rovib_row.addWidget(self.qm_invert_imaginary)
            rovib_row.addWidget(self.qm_exclude_modes)
            rovib_row.addWidget(log_summary_button)
            rovib_row.addWidget(promote_rovib_button)
            layout.addLayout(rovib_row)

            electronic_row = QHBoxLayout()
            electronic_row.addWidget(QLabel("Electronic log"))
            self.qm_electronic_log_path = QLineEdit()
            electronic_log_browse = QPushButton("Browse")
            electronic_log_browse.clicked.connect(self.browse_qm_electronic_log_path)
            self.qm_electronic_orbital_file = QLineEdit()
            self.qm_electronic_orbital_file.setPlaceholderText("optional Molden/Cube/FCHK")
            electronic_orbital_browse = QPushButton("Orbital File")
            electronic_orbital_browse.clicked.connect(self.browse_qm_electronic_orbital_file)
            self.qm_write_electronic = QCheckBox("#ELECTRONIC")
            self.qm_write_electronic.setChecked(True)
            self.qm_write_transitions = QCheckBox("#TRANSITIONS")
            self.qm_write_transitions.setChecked(True)
            promote_electronic_button = QPushButton("Promote Electronic")
            promote_electronic_button.clicked.connect(self.run_qm_promote_electronic)
            electronic_row.addWidget(self.qm_electronic_log_path, stretch=1)
            electronic_row.addWidget(electronic_log_browse)
            electronic_row.addWidget(self.qm_electronic_orbital_file, stretch=1)
            electronic_row.addWidget(electronic_orbital_browse)
            electronic_row.addWidget(self.qm_write_electronic)
            electronic_row.addWidget(self.qm_write_transitions)
            electronic_row.addWidget(promote_electronic_button)
            layout.addLayout(electronic_row)

            external_row = QHBoxLayout()
            external_row.addWidget(QLabel("QM output"))
            self.qm_external_kind = QComboBox()
            self.qm_external_kind.addItems(("molpro", "mrcc", "orca"))
            self.qm_external_output = QLineEdit()
            external_browse = QPushButton("Browse")
            external_browse.clicked.connect(self.browse_qm_external_output)
            external_summary_button = QPushButton("Summary")
            external_summary_button.clicked.connect(self.run_qm_external_summary)
            external_promote_button = QPushButton("Promote")
            external_promote_button.clicked.connect(self.run_qm_external_promote)
            external_row.addWidget(self.qm_external_kind)
            external_row.addWidget(self.qm_external_output, stretch=1)
            external_row.addWidget(external_browse)
            external_row.addWidget(external_summary_button)
            external_row.addWidget(external_promote_button)
            layout.addLayout(external_row)

            self.qm_jobs_summary = QTextEdit()
            self.qm_jobs_summary.setReadOnly(True)
            self.qm_jobs_summary.setMaximumHeight(120)
            layout.addWidget(self.qm_jobs_summary)

            self.qm_jobs_table_tabs = QTabWidget()
            self.qm_jobs_section_table = QTableWidget(0, 3)
            self.qm_jobs_action_table = QTableWidget(0, 6)
            self.qm_jobs_capability_table = QTableWidget(0, 1)
            self.qm_jobs_export_table = QTableWidget(0, 2)
            self.qm_jobs_table_tabs.addTab(self.qm_jobs_section_table, "Sections")
            self.qm_jobs_table_tabs.addTab(self.qm_jobs_action_table, "Actions")
            self.qm_jobs_table_tabs.addTab(self.qm_jobs_capability_table, "Capabilities")
            self.qm_jobs_table_tabs.addTab(self.qm_jobs_export_table, "Viewers")
            layout.addWidget(self.qm_jobs_table_tabs, stretch=1)
            return tab

        def browse_qm_gaussian_input_output(self) -> None:
            path, _selected = QFileDialog.getSaveFileName(
                self,
                "Write Gaussian GIC input",
                self.qm_gaussian_input_output.text().strip() or str(Path.cwd() / "oracle.gjf"),
                "Gaussian input (*.gjf *.com);;All files (*)",
            )
            if path:
                self.qm_gaussian_input_output.setText(path)

        def browse_qm_gaussian_workdir(self) -> None:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select Gaussian job directory",
                self.qm_gaussian_workdir.text().strip() or str(Path.cwd()),
            )
            if path:
                self.qm_gaussian_workdir.setText(path)

        def browse_qm_gaussian_job_input(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select Gaussian input",
                self.qm_gaussian_job_input.text().strip() or str(Path.cwd()),
                "Gaussian input (*.gjf *.com *.inp);;All files (*)",
            )
            if path:
                self.qm_gaussian_job_input.setText(path)

        def browse_qm_chk_path(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select Gaussian checkpoint",
                self.qm_chk_path.text().strip() or str(Path.cwd()),
                "Gaussian checkpoint (*.chk);;All files (*)",
            )
            if not path:
                return
            chk = Path(path)
            self.qm_chk_path.setText(str(chk))
            if not self.qm_fchk_output.text().strip():
                self.qm_fchk_output.setText(str(default_qm_formchk_output(chk)))

        def browse_qm_fchk_output(self) -> None:
            path, _selected = QFileDialog.getSaveFileName(
                self,
                "Write Gaussian FCHK",
                self.qm_fchk_output.text().strip() or str(Path.cwd() / "job.fchk"),
                "Gaussian FCHK (*.fchk *.fch);;All files (*)",
            )
            if path:
                self.qm_fchk_output.setText(path)
                if not self.qm_fchk_path.text().strip():
                    self.qm_fchk_path.setText(path)

        def browse_qm_fchk_path(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select Gaussian FCHK",
                self.qm_fchk_path.text().strip() or str(Path.cwd()),
                "Gaussian FCHK (*.fchk *.fch);;All files (*)",
            )
            if path:
                self.qm_fchk_path.setText(path)

        def browse_qm_log_path(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select Gaussian log/output",
                self.qm_log_path.text().strip() or str(Path.cwd()),
                "Gaussian log (*.log *.out);;All files (*)",
            )
            if path:
                self.qm_log_path.setText(path)
                if (
                    hasattr(self, "qm_electronic_log_path")
                    and not self.qm_electronic_log_path.text().strip()
                ):
                    self.qm_electronic_log_path.setText(path)

        def browse_qm_electronic_log_path(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select Gaussian electronic log/output",
                self.qm_electronic_log_path.text().strip() or str(Path.cwd()),
                "Gaussian log (*.log *.out);;All files (*)",
            )
            if path:
                self.qm_electronic_log_path.setText(path)

        def browse_qm_electronic_orbital_file(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select orbital or density file",
                self.qm_electronic_orbital_file.text().strip() or str(Path.cwd()),
                "Orbital/density files (*.molden *.molden.input *.cube *.cub *.fchk *.fch);;All files (*)",
            )
            if path:
                self.qm_electronic_orbital_file.setText(path)

        def browse_qm_external_output(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select QM output",
                self.qm_external_output.text().strip() or str(Path.cwd()),
                "QM output (*.out *.log);;All files (*)",
            )
            if path:
                self.qm_external_output.setText(path)

        def browse_qm_remote_input(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select remote QM input",
                self.qm_remote_input.text().strip() or str(Path.cwd()),
                "QM input (*.gjf *.com *.inp);;All files (*)",
            )
            if path:
                self.qm_remote_input.setText(path)

        def browse_qm_remote_destination(self) -> None:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select remote fetch destination",
                self.qm_remote_destination.text().strip() or str(Path.cwd()),
            )
            if path:
                self.qm_remote_destination.setText(path)

        def run_qm_gaussian_input(self) -> None:
            if not self._ensure_qm_project("Gaussian input"):
                return
            output = self._qm_required_path(
                self.qm_gaussian_input_output,
                "Gaussian input",
                "Gaussian input output",
            )
            if output is None:
                return
            route = self.qm_gaussian_route.text().strip() or "#p hf/sto-3g opt=readallgic"
            command = self.qm_jobs_controller.gaussian_input_command(output, route=route)
            self._start_command(command, command.label)

        def run_qm_gaussian_status(self) -> None:
            if not self._ensure_qm_idle("Gaussian status"):
                return
            workdir = self._qm_required_path(
                self.qm_gaussian_workdir,
                "Gaussian status",
                "Gaussian job directory",
            )
            if workdir is None:
                return
            command = self.qm_jobs_controller.gaussian_status_command(workdir)
            self._start_command(command, command.label)

        def run_qm_gaussian_run(self) -> None:
            if not self._ensure_qm_idle("Gaussian run"):
                return
            workdir = self._qm_required_path(
                self.qm_gaussian_workdir,
                "Gaussian run",
                "Gaussian job directory",
            )
            if workdir is None:
                return
            timeout = self._optional_float_field(
                self.qm_gaussian_timeout,
                "Gaussian run",
                "timeout",
            )
            if timeout is False:
                return
            command = self.qm_jobs_controller.gaussian_run_command(
                workdir,
                executable=self._qm_optional_text(self.qm_gaussian_executable),
                input_path=self._qm_optional_path(self.qm_gaussian_job_input),
                background=self.qm_gaussian_background.isChecked(),
                timeout=timeout,
            )
            self._start_command(command, command.label)

        def run_qm_remote_submit(self) -> None:
            if not self._ensure_qm_idle("Remote QM submit"):
                return
            input_path = self._qm_required_path(
                self.qm_remote_input,
                "Remote QM submit",
                "remote QM input",
            )
            if input_path is None:
                return
            command = self.qm_jobs_controller.remote_submit_command(
                input_path,
                engine=self.qm_remote_engine.currentText(),
                host=self.qm_remote_host.currentText().strip(),
            )
            self._start_command(command, command.label)

        def run_qm_remote_status(self) -> None:
            if not self._ensure_qm_idle("Remote QM status"):
                return
            command = self.qm_jobs_controller.remote_status_command(
                host=self.qm_remote_host.currentText().strip()
            )
            self._start_command(command, command.label)

        def run_qm_remote_fetch(self) -> None:
            promote = self.qm_remote_promote.currentText()
            if promote == "none":
                if not self._ensure_qm_idle("Remote QM fetch"):
                    return
                target_xyzin = None
            else:
                if not self._ensure_qm_project("Remote QM fetch"):
                    return
                target_xyzin = self.controller.xyzin
            job = self.qm_remote_job.text().strip()
            if not job:
                QMessageBox.warning(self, "Remote QM fetch", "Enter a remote job id first.")
                return
            command = self.qm_jobs_controller.remote_fetch_command(
                job,
                host=self.qm_remote_host.currentText().strip(),
                destination=self.qm_remote_destination.text().strip() or "remote_qm_runs",
                promote=promote,
                xyzin=target_xyzin,
            )
            if target_xyzin is not None:
                self.pending_xyzin_after_run = target_xyzin
            self._start_command(command, command.label)

        def run_qm_gaussian_formchk(self) -> None:
            if not self._ensure_qm_idle("Gaussian formchk"):
                return
            chk = self._qm_required_path(self.qm_chk_path, "Gaussian formchk", "CHK file")
            if chk is None:
                return
            fchk = self._qm_optional_path(self.qm_fchk_output)
            if fchk is None:
                fchk = default_qm_formchk_output(chk)
                self.qm_fchk_output.setText(str(fchk))
            timeout = self._optional_float_field(
                self.qm_formchk_timeout,
                "Gaussian formchk",
                "timeout",
            )
            if timeout is False:
                return
            self.qm_fchk_path.setText(str(fchk))
            command = self.qm_jobs_controller.gaussian_formchk_command(
                chk,
                fchk,
                executable=self._qm_optional_text(self.qm_formchk_executable),
                timeout=timeout,
            )
            self._start_command(command, command.label)

        def run_qm_fchk_summary(self) -> None:
            if not self._ensure_qm_idle("Gaussian FCHK summary"):
                return
            fchk = self._qm_required_path(
                self.qm_fchk_path,
                "Gaussian FCHK summary",
                "FCHK file",
            )
            if fchk is None:
                return
            command = self.qm_jobs_controller.gaussian_fchk_summary_command(fchk)
            self._start_command(command, command.label)

        def run_qm_promote_fchk(self) -> None:
            if not self._ensure_qm_project("Promote Gaussian FCHK"):
                return
            fchk = self._qm_required_path(
                self.qm_fchk_path,
                "Promote Gaussian FCHK",
                "FCHK file",
            )
            if fchk is None:
                return
            command = self.qm_jobs_controller.gaussian_promote_fchk_command(
                fchk,
                cartesian_hessian=self.qm_write_cartesian_hessian.isChecked(),
                normal_modes=self.qm_write_normal_modes.isChecked(),
                qff=self.qm_write_qff.isChecked(),
                electronic=self.qm_write_fchk_electronic.isChecked(),
                orbitals=self.qm_write_fchk_orbitals.isChecked(),
            )
            self.pending_xyzin_after_run = self.controller.xyzin
            self._start_command(command, command.label)

        def run_qm_log_summary(self) -> None:
            if not self._ensure_qm_idle("Gaussian log summary"):
                return
            log = self._qm_required_path(self.qm_log_path, "Gaussian log summary", "Gaussian log")
            if log is None:
                return
            command = self.qm_jobs_controller.gaussian_log_summary_command(log)
            self._start_command(command, command.label)

        def run_qm_promote_rovib(self) -> None:
            if not self._ensure_qm_project("Promote Gaussian rovib"):
                return
            log = self._qm_required_path(self.qm_log_path, "Promote Gaussian rovib", "Gaussian log")
            if log is None:
                return
            exclude_modes = self._qm_exclude_modes()
            if exclude_modes is None:
                return
            command = self.qm_jobs_controller.gaussian_promote_rovib_command(
                log,
                vibrational=self.qm_write_vibrational.isChecked(),
                rotational=self.qm_write_rotational.isChecked(),
                deltabvib=self.qm_write_deltabvib.isChecked(),
                invert_imaginary=self.qm_invert_imaginary.isChecked(),
                exclude_modes=exclude_modes,
            )
            self.pending_xyzin_after_run = self.controller.xyzin
            self._start_command(command, command.label)

        def run_qm_promote_electronic(self) -> None:
            if not self._ensure_qm_project("Promote Gaussian electronic data"):
                return
            log = self._qm_required_path(
                self.qm_electronic_log_path,
                "Promote Gaussian electronic data",
                "Gaussian electronic log",
            )
            if log is None:
                return
            orbital_file = self._qm_optional_path(self.qm_electronic_orbital_file)
            command = self.qm_jobs_controller.gaussian_promote_electronic_command(
                log,
                electronic=self.qm_write_electronic.isChecked(),
                transitions=self.qm_write_transitions.isChecked(),
                orbital_files=() if orbital_file is None else (orbital_file,),
            )
            self.pending_xyzin_after_run = self.controller.xyzin
            self._start_command(command, command.label)

        def run_qm_external_summary(self) -> None:
            if not self._ensure_qm_idle("QM output summary"):
                return
            output = self._qm_required_path(
                self.qm_external_output,
                "QM output summary",
                "QM output file",
            )
            if output is None:
                return
            if self.qm_external_kind.currentText() == "molpro":
                command = self.qm_jobs_controller.molpro_summary_command(output)
            elif self.qm_external_kind.currentText() == "mrcc":
                command = self.qm_jobs_controller.mrcc_summary_command(output)
            else:
                command = self.qm_jobs_controller.orca_summary_command(output)
            self._start_command(command, command.label)

        def run_qm_external_promote(self) -> None:
            if not self._ensure_qm_project("Promote QM output"):
                return
            output = self._qm_required_path(
                self.qm_external_output,
                "Promote QM output",
                "QM output file",
            )
            if output is None:
                return
            if self.qm_external_kind.currentText() == "molpro":
                command = self.qm_jobs_controller.molpro_promote_command(output)
            elif self.qm_external_kind.currentText() == "mrcc":
                command = self.qm_jobs_controller.mrcc_promote_command(output)
            else:
                command = self.qm_jobs_controller.orca_promote_command(output)
            self.pending_xyzin_after_run = self.controller.xyzin
            self._start_command(command, command.label)

        def _ensure_qm_project(self, title: str) -> bool:
            if self.controller.xyzin is None:
                QMessageBox.warning(self, title, "Open or preprocess a MATRIX xyzin first.")
                return False
            if not self._ensure_qm_idle(title):
                return False
            self.qm_jobs_controller.set_xyzin(self.controller.xyzin)
            return True

        def _ensure_qm_idle(self, title: str) -> bool:
            if self.process is not None:
                QMessageBox.information(self, "ORACLE", "A command is already running.")
                return False
            return True

        def _qm_required_path(
            self,
            field: QLineEdit,
            title: str,
            label: str,
        ) -> Path | None:
            text = field.text().strip()
            if not text:
                QMessageBox.warning(self, title, f"Select {label} first.")
                return None
            return Path(text)

        def _qm_optional_path(self, field: QLineEdit) -> Path | None:
            text = field.text().strip()
            return Path(text) if text else None

        def _qm_optional_text(self, field: QLineEdit) -> str | None:
            text = field.text().strip()
            return text or None

        def _qm_exclude_modes(self) -> tuple[int, ...] | None:
            text = self.qm_exclude_modes.text().strip()
            if not text:
                return ()
            modes: list[int] = []
            for token in text.replace(",", " ").replace(";", " ").split():
                try:
                    value = int(token)
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Promote Gaussian rovib",
                        f"Invalid excluded mode index: {token}",
                    )
                    return None
                if value <= 0:
                    QMessageBox.warning(
                        self,
                        "Promote Gaussian rovib",
                        "Excluded mode indices must be positive.",
                    )
                    return None
                modes.append(value)
            return tuple(modes)

        def _set_default_qm_jobs_outputs(self, xyzin: Path) -> None:
            self.qm_gaussian_input_output.setText(str(default_qm_gaussian_input_output(xyzin)))
            self.qm_gaussian_workdir.setText(str(default_qm_gaussian_workdir(xyzin)))
            if not self.qm_gaussian_job_input.text().strip():
                self.qm_gaussian_job_input.setText(str(default_qm_gaussian_input_output(xyzin)))

        def _populate_qm_jobs_tables(self, xyzin: Path) -> None:
            state = load_workbench_gui_state(xyzin, "qm_jobs")
            self.qm_jobs_summary.setPlainText("\n".join(workbench_gui_state_lines(state)))
            self._fill_table(self.qm_jobs_section_table, state.sections)
            self._fill_table(self.qm_jobs_action_table, state.actions)
            self._fill_table(self.qm_jobs_capability_table, state.capabilities)
            self._fill_table(self.qm_jobs_export_table, state.exports)

        def _clear_qm_jobs_tables(self) -> None:
            summary = getattr(self, "qm_jobs_summary", None)
            if summary is not None:
                summary.clear()
            for table in (
                getattr(self, "qm_jobs_section_table", None),
                getattr(self, "qm_jobs_action_table", None),
                getattr(self, "qm_jobs_capability_table", None),
                getattr(self, "qm_jobs_export_table", None),
            ):
                if table is not None:
                    table.setRowCount(0)

        def _build_electronic_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            file_row = QHBoxLayout()
            file_row.addWidget(QLabel("Orbital / density file"))
            self.electronic_viewer_target = QLineEdit()
            browse_button = QPushButton("Browse")
            browse_button.clicked.connect(self.browse_electronic_viewer_target)
            file_row.addWidget(self.electronic_viewer_target, stretch=1)
            file_row.addWidget(browse_button)
            layout.addLayout(file_row)

            viewer_row = QHBoxLayout()
            self.electronic_molden_executable = QLineEdit("molden")
            self.electronic_avogadro_executable = QLineEdit("avogadro2")
            self.electronic_morbvis_url = QLineEdit(MORBVIS_URL)
            molden_button = QPushButton("Molden")
            molden_button.clicked.connect(self.run_electronic_molden)
            avogadro_button = QPushButton("Avogadro")
            avogadro_button.clicked.connect(self.run_electronic_avogadro)
            morbvis_button = QPushButton("MOrbVis")
            morbvis_button.clicked.connect(self.run_electronic_morbvis)
            selected_button = QPushButton("Open Selected")
            selected_button.clicked.connect(self.run_electronic_selected_viewer)
            viewer_row.addWidget(QLabel("Molden exe"))
            viewer_row.addWidget(self.electronic_molden_executable)
            viewer_row.addWidget(QLabel("Avogadro exe"))
            viewer_row.addWidget(self.electronic_avogadro_executable)
            viewer_row.addWidget(QLabel("MOrbVis URL"))
            viewer_row.addWidget(self.electronic_morbvis_url)
            viewer_row.addWidget(molden_button)
            viewer_row.addWidget(avogadro_button)
            viewer_row.addWidget(morbvis_button)
            viewer_row.addWidget(selected_button)
            layout.addLayout(viewer_row)

            export_row = QHBoxLayout()
            export_row.addWidget(QLabel("Publication"))
            self.electronic_export_dir = QLineEdit()
            export_browse = QPushButton("Browse")
            export_browse.clicked.connect(self.browse_electronic_export_dir)
            self.electronic_export_csv = QCheckBox("CSV")
            self.electronic_export_csv.setChecked(True)
            self.electronic_export_svg = QCheckBox("SVG")
            self.electronic_export_svg.setChecked(True)
            self.electronic_export_pdf = QCheckBox("PDF")
            self.electronic_export_pdf.setChecked(True)
            export_button = QPushButton("Export Spectrum")
            export_button.clicked.connect(self.run_electronic_publication_export)
            export_row.addWidget(self.electronic_export_dir, stretch=1)
            export_row.addWidget(export_browse)
            export_row.addWidget(self.electronic_export_csv)
            export_row.addWidget(self.electronic_export_svg)
            export_row.addWidget(self.electronic_export_pdf)
            export_row.addWidget(export_button)
            layout.addLayout(export_row)

            self.electronic_summary = QTextEdit()
            self.electronic_summary.setReadOnly(True)
            self.electronic_summary.setMaximumHeight(120)
            layout.addWidget(self.electronic_summary)

            self.electronic_table_tabs = QTabWidget()
            self.electronic_section_table = QTableWidget(0, 3)
            self.electronic_transition_table = QTableWidget(0, 2)
            self.electronic_orbital_table = QTableWidget(0, 2)
            self.electronic_table_tabs.addTab(self.electronic_section_table, "Sections")
            self.electronic_table_tabs.addTab(self.electronic_transition_table, "Transitions")
            self.electronic_table_tabs.addTab(self.electronic_orbital_table, "Orbitals")
            layout.addWidget(self.electronic_table_tabs, stretch=1)
            return tab

        def browse_electronic_viewer_target(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select orbital or density file",
                self.electronic_viewer_target.text().strip() or str(Path.cwd()),
                "Orbital/density files (*.molden *.molden.input *.fchk *.fch *.cube *.cub *.wfx *.wfn);;All files (*)",
            )
            if path:
                self.electronic_viewer_target.setText(path)

        def run_electronic_molden(self) -> None:
            if not self._ensure_electronic_idle("Molden"):
                return
            target = self._qm_required_path(self.electronic_viewer_target, "Molden", "viewer file")
            if target is None:
                return
            executable = self.electronic_molden_executable.text().strip() or "molden"
            command = self.electronic_controller.molden_command(target, executable=executable)
            self._start_command(command, command.label)

        def run_electronic_avogadro(self) -> None:
            if not self._ensure_electronic_idle("Avogadro"):
                return
            target = self._qm_required_path(
                self.electronic_viewer_target, "Avogadro", "viewer file"
            )
            if target is None:
                return
            executable = self.electronic_avogadro_executable.text().strip() or "avogadro2"
            command = self.electronic_controller.avogadro_command(target, executable=executable)
            self._start_command(command, command.label)

        def run_electronic_morbvis(self) -> None:
            if not self._ensure_electronic_idle("MOrbVis"):
                return
            url = self.electronic_morbvis_url.text().strip() or MORBVIS_URL
            command = self.electronic_controller.morbvis_command(url=url)
            self._start_command(command, command.label)

        def run_electronic_selected_viewer(self) -> None:
            if not self._ensure_electronic_idle("Open selected orbital"):
                return
            row = self.electronic_orbital_table.currentRow()
            try:
                command = self.electronic_controller.selected_orbital_viewer_command(
                    row,
                    molden_executable=self.electronic_molden_executable.text().strip() or "molden",
                    avogadro_executable=self.electronic_avogadro_executable.text().strip()
                    or "avogadro2",
                    morbvis_url=self.electronic_morbvis_url.text().strip() or MORBVIS_URL,
                )
            except (FileNotFoundError, ValueError) as exc:
                QMessageBox.warning(self, "Open selected orbital", str(exc))
                return
            self._start_command(command, command.label)

        def browse_electronic_export_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select electronic publication export directory",
                self.electronic_export_dir.text().strip() or str(Path.cwd()),
            )
            if path:
                self.electronic_export_dir.setText(path)

        def run_electronic_publication_export(self) -> None:
            if self.controller.xyzin is None:
                QMessageBox.warning(
                    self, "Electronic publication export", "Open a MATRIX xyzin first."
                )
                return
            formats = self._electronic_export_formats()
            if not formats:
                QMessageBox.warning(
                    self,
                    "Electronic publication export",
                    "Select at least one export format.",
                )
                return
            outdir = (
                Path(self.electronic_export_dir.text().strip())
                if self.electronic_export_dir.text().strip()
                else default_electronic_export_dir(self.controller.xyzin)
            )
            try:
                result = self.electronic_controller.export_electronic_publication(
                    outdir, formats=formats
                )
            except ValueError as exc:
                QMessageBox.warning(self, "Electronic publication export", str(exc))
                return
            QMessageBox.information(
                self,
                "Electronic publication export",
                "Wrote Electronic publication export:\n"
                + "\n".join(str(path) for path in result.paths),
            )

        def _ensure_electronic_idle(self, title: str) -> bool:
            if self.process is not None:
                QMessageBox.information(self, "ORACLE", "A command is already running.")
                return False
            if self.controller.xyzin is not None:
                self.electronic_controller.set_xyzin(self.controller.xyzin)
            return True

        def _populate_electronic_tables(self, xyzin: Path) -> None:
            state = load_electronic_gui_state(xyzin)
            self.electronic_summary.setPlainText("\n".join(electronic_gui_state_lines(state)))
            self._fill_table(self.electronic_section_table, state.sections)
            self._fill_table(self.electronic_transition_table, state.transitions)
            self._fill_table(self.electronic_orbital_table, state.orbitals)

        def _clear_electronic_tables(self) -> None:
            summary = getattr(self, "electronic_summary", None)
            if summary is not None:
                summary.clear()
            for table in (
                getattr(self, "electronic_section_table", None),
                getattr(self, "electronic_transition_table", None),
                getattr(self, "electronic_orbital_table", None),
            ):
                if table is not None:
                    table.setRowCount(0)

        def _electronic_export_formats(self) -> tuple[str, ...]:
            formats: list[str] = []
            if self.electronic_export_csv.isChecked():
                formats.append("csv")
            if self.electronic_export_svg.isChecked():
                formats.append("svg")
            if self.electronic_export_pdf.isChecked():
                formats.append("pdf")
            return tuple(formats)

        def _set_default_electronic_outputs(self, xyzin: Path) -> None:
            self.electronic_export_dir.setText(str(default_electronic_export_dir(xyzin)))

        def _build_sefit_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            job_row = QHBoxLayout()
            job_row.addWidget(QLabel("Job"))
            self.sefit_job_path = QLineEdit()
            job_browse = QPushButton("Browse")
            job_browse.clicked.connect(self.browse_sefit_job)
            job_row.addWidget(self.sefit_job_path, stretch=1)
            job_row.addWidget(job_browse)
            layout.addLayout(job_row)

            out_row = QHBoxLayout()
            out_row.addWidget(QLabel("Outdir"))
            self.sefit_outdir = QLineEdit()
            out_browse = QPushButton("Browse")
            out_browse.clicked.connect(self.browse_sefit_outdir)
            self.sefit_backend = QComboBox()
            self.sefit_backend.addItems(("python", "fortran77"))
            self.sefit_write_section = QCheckBox("Update #MORPHEUS")
            self.sefit_write_section.setChecked(True)
            out_row.addWidget(self.sefit_outdir, stretch=1)
            out_row.addWidget(out_browse)
            out_row.addWidget(QLabel("Backend"))
            out_row.addWidget(self.sefit_backend)
            out_row.addWidget(self.sefit_write_section)
            layout.addLayout(out_row)

            option_row = QHBoxLayout()
            self.sefit_extra_args = QLineEdit()
            self.sefit_extra_args.setPlaceholderText("--max-iter 20 --coordinate-model gic")
            run_button = QPushButton("Run SEFit")
            run_button.clicked.connect(self.run_sefit)
            option_row.addWidget(QLabel("Options"))
            option_row.addWidget(self.sefit_extra_args, stretch=1)
            option_row.addWidget(run_button)
            layout.addLayout(option_row)

            self.sefit_summary = QTextEdit()
            self.sefit_summary.setReadOnly(True)
            self.sefit_summary.setMaximumHeight(150)
            layout.addWidget(self.sefit_summary)

            self.sefit_table_tabs = QTabWidget()
            self.sefit_isotopologue_table = QTableWidget(0, 6)
            self.sefit_output_table = QTableWidget(0, 2)
            self.sefit_diagnostics_table = QTableWidget(0, 2)
            self.sefit_table_tabs.addTab(self.sefit_isotopologue_table, "Isotopologues")
            self.sefit_table_tabs.addTab(self.sefit_output_table, "Outputs")
            self.sefit_table_tabs.addTab(self.sefit_diagnostics_table, "Diagnostics")
            layout.addWidget(self.sefit_table_tabs, stretch=1)
            return tab

        def browse_sefit_job(self) -> None:
            path, _selected = QFileDialog.getOpenFileName(
                self,
                "Select SEFit / MORPHEUS job",
                str(Path.cwd()),
                "SEFit jobs (*.toml *.mfit *.msr *.inp);;All files (*)",
            )
            if path:
                self.sefit_job_path.setText(path)

        def browse_sefit_outdir(self) -> None:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select SEFit output directory",
                self.sefit_outdir.text().strip() or str(Path.cwd()),
            )
            if path:
                self.sefit_outdir.setText(path)

        def run_sefit(self) -> None:
            if not self._ensure_sefit_ready_for_command("SEFit / MORPHEUS"):
                return
            job_text = self.sefit_job_path.text().strip()
            if not job_text:
                QMessageBox.warning(self, "SEFit / MORPHEUS", "Select a SEFit job first.")
                return
            self.sefit_controller.set_xyzin(self.controller.xyzin)
            command = self.sefit_controller.run_command(
                job=Path(job_text),
                outdir=self._sefit_outdir(),
                backend=self.sefit_backend.currentText(),
                write_section=self.sefit_write_section.isChecked(),
                extra_args=self._sefit_extra_args(),
            )
            self._start_command(command, command.label)

        def _ensure_sefit_ready_for_command(self, title: str) -> bool:
            if self.controller.xyzin is None:
                QMessageBox.warning(self, title, "Open or preprocess a MATRIX xyzin first.")
                return False
            if self.process is not None:
                QMessageBox.information(self, "ORACLE", "A command is already running.")
                return False
            self.sefit_controller.set_xyzin(self.controller.xyzin)
            return True

        def _sefit_outdir(self) -> Path:
            if self.controller.xyzin is None:
                raise ValueError("no MATRIX xyzin project is loaded")
            text = self.sefit_outdir.text().strip()
            outdir = Path(text) if text else default_sefit_outdir(self.controller.xyzin)
            self.sefit_outdir.setText(str(outdir))
            return outdir

        def _sefit_extra_args(self) -> tuple[str, ...]:
            import shlex

            return tuple(shlex.split(self.sefit_extra_args.text().strip()))

        def _set_default_sefit_outputs(self, xyzin: Path) -> None:
            self.sefit_outdir.setText(str(default_sefit_outdir(xyzin)))

        def _populate_sefit_tables(self, xyzin: Path) -> None:
            state = load_sefit_gui_state(xyzin)
            self.sefit_summary.setPlainText("\n".join(sefit_gui_state_lines(state)))
            self._fill_table(self.sefit_isotopologue_table, state.isotopologues)
            self._fill_table(self.sefit_output_table, state.outputs)
            self._fill_table(self.sefit_diagnostics_table, state.diagnostics)

        def _clear_sefit_tables(self) -> None:
            summary = getattr(self, "sefit_summary", None)
            if summary is not None:
                summary.clear()
            for table in (
                getattr(self, "sefit_isotopologue_table", None),
                getattr(self, "sefit_output_table", None),
                getattr(self, "sefit_diagnostics_table", None),
            ):
                if table is not None:
                    table.setRowCount(0)

        def _build_thermo_kinetics_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            thermo_row = QHBoxLayout()
            thermo_row.addWidget(QLabel("Thermo report"))
            self.thermo_report_output = QLineEdit()
            report_browse = QPushButton("Save As")
            report_browse.clicked.connect(self.browse_thermo_report_output)
            self.thermo_cutoff_cm1 = QLineEdit("10.0")
            self.thermo_write_report = QCheckBox("Report")
            self.thermo_write_report.setChecked(True)
            self.thermo_write_section = QCheckBox("Update #THERMO")
            self.thermo_write_section.setChecked(True)
            self.thermo_keep_low_positive = QCheckBox("Keep low positive")
            summary_button = QPushButton("Rovib Summary")
            summary_button.clicked.connect(self.run_thermo_rovib_summary)
            run_button = QPushButton("Run Thermo")
            run_button.clicked.connect(self.run_thermo)
            thermo_row.addWidget(self.thermo_report_output, stretch=1)
            thermo_row.addWidget(report_browse)
            thermo_row.addWidget(QLabel("Cutoff"))
            thermo_row.addWidget(self.thermo_cutoff_cm1)
            thermo_row.addWidget(self.thermo_write_report)
            thermo_row.addWidget(self.thermo_write_section)
            thermo_row.addWidget(self.thermo_keep_low_positive)
            thermo_row.addWidget(summary_button)
            thermo_row.addWidget(run_button)
            layout.addLayout(thermo_row)

            dos_files_row = QHBoxLayout()
            dos_files_row.addWidget(QLabel("Vib DOS"))
            self.thermo_vib_dos_output = QLineEdit()
            vib_dos_browse = QPushButton("Save As")
            vib_dos_browse.clicked.connect(self.browse_thermo_vib_dos_output)
            dos_files_row.addWidget(self.thermo_vib_dos_output, stretch=1)
            dos_files_row.addWidget(vib_dos_browse)
            dos_files_row.addWidget(QLabel("Rovib DOS"))
            self.thermo_rovib_dos_output = QLineEdit()
            rovib_dos_browse = QPushButton("Save As")
            rovib_dos_browse.clicked.connect(self.browse_thermo_rovib_dos_output)
            dos_files_row.addWidget(self.thermo_rovib_dos_output, stretch=1)
            dos_files_row.addWidget(rovib_dos_browse)
            layout.addLayout(dos_files_row)

            dos_extra_row = QHBoxLayout()
            dos_extra_row.addWidget(QLabel("Rot DOS"))
            self.thermo_rot_dos_output = QLineEdit()
            rot_dos_browse = QPushButton("Save As")
            rot_dos_browse.clicked.connect(self.browse_thermo_rot_dos_output)
            dos_extra_row.addWidget(self.thermo_rot_dos_output, stretch=1)
            dos_extra_row.addWidget(rot_dos_browse)
            dos_extra_row.addWidget(QLabel("Q(T)"))
            self.thermo_rovib_q_output = QLineEdit()
            q_browse = QPushButton("Save As")
            q_browse.clicked.connect(self.browse_thermo_rovib_q_output)
            dos_extra_row.addWidget(self.thermo_rovib_q_output, stretch=1)
            dos_extra_row.addWidget(q_browse)
            layout.addLayout(dos_extra_row)

            dos_options_row = QHBoxLayout()
            self.thermo_vmax = QLineEdit("6")
            self.thermo_emax_cm1 = QLineEdit("8000.0")
            self.thermo_bin_cm1 = QLineEdit("50.0")
            self.thermo_ncap = QLineEdit("10.0")
            self.thermo_emax_rot = QLineEdit()
            self.thermo_emax_rot.setPlaceholderText("rot Emax")
            self.thermo_jmax = QLineEdit()
            self.thermo_jmax.setPlaceholderText("Jmax")
            vib_dos_button = QPushButton("Build Vib DOS")
            vib_dos_button.clicked.connect(self.run_thermo_vibrational_dos)
            rovib_dos_button = QPushButton("Build Rovib DOS")
            rovib_dos_button.clicked.connect(self.run_thermo_rovib_dos)
            dos_options_row.addWidget(QLabel("vmax"))
            dos_options_row.addWidget(self.thermo_vmax)
            dos_options_row.addWidget(QLabel("Emax"))
            dos_options_row.addWidget(self.thermo_emax_cm1)
            dos_options_row.addWidget(QLabel("bin"))
            dos_options_row.addWidget(self.thermo_bin_cm1)
            dos_options_row.addWidget(QLabel("ncap"))
            dos_options_row.addWidget(self.thermo_ncap)
            dos_options_row.addWidget(self.thermo_emax_rot)
            dos_options_row.addWidget(self.thermo_jmax)
            dos_options_row.addWidget(vib_dos_button)
            dos_options_row.addWidget(rovib_dos_button)
            layout.addLayout(dos_options_row)

            export_row = QHBoxLayout()
            export_row.addWidget(QLabel("Publication"))
            self.thermo_export_dir = QLineEdit()
            export_browse = QPushButton("Browse")
            export_browse.clicked.connect(self.browse_thermo_export_dir)
            self.thermo_export_csv = QCheckBox("CSV")
            self.thermo_export_csv.setChecked(True)
            self.thermo_export_tex = QCheckBox("LaTeX")
            self.thermo_export_tex.setChecked(True)
            self.thermo_export_svg = QCheckBox("SVG")
            self.thermo_export_svg.setChecked(True)
            self.thermo_export_pdf = QCheckBox("PDF")
            self.thermo_export_pdf.setChecked(True)
            export_button = QPushButton("Export Thermo")
            export_button.clicked.connect(self.run_thermo_publication_export)
            export_row.addWidget(self.thermo_export_dir, stretch=1)
            export_row.addWidget(export_browse)
            export_row.addWidget(self.thermo_export_csv)
            export_row.addWidget(self.thermo_export_tex)
            export_row.addWidget(self.thermo_export_svg)
            export_row.addWidget(self.thermo_export_pdf)
            export_row.addWidget(export_button)
            layout.addLayout(export_row)

            self.thermo_kinetics_summary = QTextEdit()
            self.thermo_kinetics_summary.setReadOnly(True)
            self.thermo_kinetics_summary.setMaximumHeight(120)
            layout.addWidget(self.thermo_kinetics_summary)

            self.thermo_kinetics_table_tabs = QTabWidget()
            self.thermo_table = QTableWidget(0, 7)
            self.thermo_dos_table = QTableWidget(0, 6)
            self.kinetics_table = QTableWidget(0, 3)
            self.thermo_kinetics_table_tabs.addTab(self.thermo_table, "Thermo")
            self.thermo_kinetics_table_tabs.addTab(self.thermo_dos_table, "DOS")
            self.thermo_kinetics_table_tabs.addTab(self.kinetics_table, "Kinetics")
            layout.addWidget(self.thermo_kinetics_table_tabs, stretch=1)
            return tab

        def browse_thermo_report_output(self) -> None:
            self._browse_thermo_save_path(
                self.thermo_report_output,
                "Write Thermo report",
                "Text reports (*.txt *.report);;All files (*)",
            )

        def browse_thermo_vib_dos_output(self) -> None:
            self._browse_thermo_save_path(
                self.thermo_vib_dos_output,
                "Write vibrational DOS",
                "DOS files (*.dat *.txt);;All files (*)",
            )

        def browse_thermo_rovib_dos_output(self) -> None:
            self._browse_thermo_save_path(
                self.thermo_rovib_dos_output,
                "Write rovibrational DOS",
                "DOS files (*.dat *.txt);;All files (*)",
            )

        def browse_thermo_rot_dos_output(self) -> None:
            self._browse_thermo_save_path(
                self.thermo_rot_dos_output,
                "Write rotational DOS",
                "DOS files (*.dat *.txt);;All files (*)",
            )

        def browse_thermo_rovib_q_output(self) -> None:
            self._browse_thermo_save_path(
                self.thermo_rovib_q_output,
                "Write rovib Q(T)",
                "Data files (*.dat *.txt);;All files (*)",
            )

        def browse_thermo_export_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select publication export directory",
                self.thermo_export_dir.text().strip() or str(Path.cwd()),
            )
            if path:
                self.thermo_export_dir.setText(path)

        def _browse_thermo_save_path(
            self,
            field: QLineEdit,
            title: str,
            file_filter: str,
        ) -> None:
            path, _selected = QFileDialog.getSaveFileName(
                self,
                title,
                field.text().strip() or str(Path.cwd()),
                file_filter,
            )
            if path:
                field.setText(path)

        def run_thermo_rovib_summary(self) -> None:
            if not self._ensure_thermo_kinetics_ready_for_command("Rovib summary"):
                return
            command = self.thermo_kinetics_controller.rovib_summary_command()
            self._start_command(command, command.label)

        def run_thermo(self) -> None:
            if not self._ensure_thermo_kinetics_ready_for_command("Thermo"):
                return
            cutoff = self._optional_float_field(
                self.thermo_cutoff_cm1,
                "Thermo",
                "cutoff",
                default=10.0,
            )
            if cutoff is False:
                return
            report_output = (
                self._optional_path_text(self.thermo_report_output)
                if self.thermo_write_report.isChecked()
                else None
            )
            command = self.thermo_kinetics_controller.thermo_command(
                out=report_output,
                report=self.thermo_write_report.isChecked(),
                write_section=self.thermo_write_section.isChecked(),
                cutoff_cm1=cutoff,
                keep_low_positive=self.thermo_keep_low_positive.isChecked(),
            )
            if self.thermo_write_section.isChecked():
                self.pending_xyzin_after_run = self.controller.xyzin
            self._start_command(command, command.label)

        def run_thermo_vibrational_dos(self) -> None:
            if not self._ensure_thermo_kinetics_ready_for_command("Vibrational DOS"):
                return
            vmax = self._required_positive_int_field(self.thermo_vmax, "Vibrational DOS", "vmax")
            if vmax is None:
                return
            emax = self._optional_float_field(
                self.thermo_emax_cm1,
                "Vibrational DOS",
                "Emax",
                default=8000.0,
            )
            if emax is False:
                return
            bin_cm1 = self._optional_float_field(
                self.thermo_bin_cm1,
                "Vibrational DOS",
                "bin",
                default=50.0,
            )
            if bin_cm1 is False:
                return
            ncap = self._optional_float_field(
                self.thermo_ncap,
                "Vibrational DOS",
                "ncap",
                default=10.0,
            )
            if ncap is False:
                return
            command = self.thermo_kinetics_controller.vibrational_dos_command(
                out=self._thermo_output_path(
                    self.thermo_vib_dos_output,
                    default_vibrational_dos_output,
                ),
                vmax=vmax,
                emax_cm1=emax,
                bin_cm1=bin_cm1,
                ncap=ncap,
            )
            self._start_command(command, command.label)

        def run_thermo_rovib_dos(self) -> None:
            if not self._ensure_thermo_kinetics_ready_for_command("Rovibrational DOS"):
                return
            emax_rot = self._optional_float_field(
                self.thermo_emax_rot,
                "Rovibrational DOS",
                "rotational Emax",
            )
            if emax_rot is False:
                return
            jmax = self._optional_positive_int_field(
                self.thermo_jmax,
                "Rovibrational DOS",
                "Jmax",
            )
            if jmax is False:
                return
            command = self.thermo_kinetics_controller.rovibrational_dos_command(
                vib_dos=self._thermo_output_path(
                    self.thermo_vib_dos_output,
                    default_vibrational_dos_output,
                ),
                out=self._thermo_output_path(
                    self.thermo_rovib_dos_output,
                    default_rovibrational_dos_output,
                ),
                rot_out=self._thermo_output_path(
                    self.thermo_rot_dos_output,
                    default_rotational_dos_output,
                ),
                q_out=self._thermo_output_path(self.thermo_rovib_q_output, default_rovib_q_output),
                emax_rot=emax_rot,
                jmax=jmax,
            )
            self._start_command(command, command.label)

        def run_thermo_publication_export(self) -> None:
            if not self._ensure_thermo_kinetics_ready_for_command("Thermo publication export"):
                return
            formats = self._thermo_publication_formats()
            if not formats:
                QMessageBox.warning(
                    self,
                    "Thermo publication export",
                    "Select at least one export format.",
                )
                return
            outdir = self._thermo_output_path(self.thermo_export_dir, default_thermo_export_dir)
            try:
                result = self.thermo_kinetics_controller.export_thermo_publication(
                    outdir,
                    formats=formats,
                )
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Thermo publication export", str(exc))
                return
            self.controller.log(
                "Wrote Thermo publication export:\n" + "\n".join(str(path) for path in result.paths)
            )
            self.log_output.setPlainText("\n".join(self.controller.log_lines))
            self.refresh()

        def _ensure_thermo_kinetics_ready_for_command(self, title: str) -> bool:
            if self.controller.xyzin is None:
                QMessageBox.warning(self, title, "Open or preprocess a MATRIX xyzin first.")
                return False
            if self.process is not None:
                QMessageBox.information(self, "ORACLE", "A command is already running.")
                return False
            self.thermo_kinetics_controller.set_xyzin(self.controller.xyzin)
            return True

        def _optional_positive_int_field(
            self,
            field: QLineEdit,
            title: str,
            label: str,
        ):
            text = field.text().strip()
            if not text:
                return None
            try:
                value = int(text)
            except ValueError:
                QMessageBox.warning(self, title, f"Invalid {label}: {text}")
                return False
            if value <= 0:
                QMessageBox.warning(self, title, f"{label} must be positive.")
                return False
            return value

        def _thermo_output_path(self, field: QLineEdit, default_factory) -> Path:
            if self.controller.xyzin is None:
                raise ValueError("no MATRIX xyzin project is loaded")
            text = field.text().strip()
            output = Path(text) if text else default_factory(self.controller.xyzin)
            field.setText(str(output))
            return output

        def _thermo_publication_formats(self) -> tuple[str, ...]:
            formats = []
            if self.thermo_export_csv.isChecked():
                formats.append("csv")
            if self.thermo_export_tex.isChecked():
                formats.append("tex")
            if self.thermo_export_svg.isChecked():
                formats.append("svg")
            if self.thermo_export_pdf.isChecked():
                formats.append("pdf")
            return tuple(formats)

        def _set_default_thermo_kinetics_outputs(self, xyzin: Path) -> None:
            self.thermo_report_output.setText(str(default_thermo_report_output(xyzin)))
            self.thermo_vib_dos_output.setText(str(default_vibrational_dos_output(xyzin)))
            self.thermo_rot_dos_output.setText(str(default_rotational_dos_output(xyzin)))
            self.thermo_rovib_dos_output.setText(str(default_rovibrational_dos_output(xyzin)))
            self.thermo_rovib_q_output.setText(str(default_rovib_q_output(xyzin)))
            self.thermo_export_dir.setText(str(default_thermo_export_dir(xyzin)))

        def _populate_thermo_kinetics_tables(self, xyzin: Path) -> None:
            state = load_thermo_kinetics_gui_state(xyzin)
            self.thermo_kinetics_summary.setPlainText(
                "\n".join(thermo_kinetics_gui_state_lines(state))
            )
            self._fill_table(self.thermo_table, state.thermo)
            self._fill_table(self.thermo_dos_table, state.dos)
            self._fill_table(self.kinetics_table, state.kinetics)

        def _clear_thermo_kinetics_tables(self) -> None:
            summary = getattr(self, "thermo_kinetics_summary", None)
            if summary is not None:
                summary.clear()
            for table in (
                getattr(self, "thermo_table", None),
                getattr(self, "thermo_dos_table", None),
                getattr(self, "kinetics_table", None),
            ):
                if table is not None:
                    table.setRowCount(0)

        def _build_link_tab(self) -> QWidget:
            tab = QWidget()
            layout = QVBoxLayout(tab)

            engine_row = QHBoxLayout()
            engine_row.addWidget(QLabel("Engine"))
            self.link_engine_command = QLineEdit()
            self.link_engine_command.setPlaceholderText(
                "external-qm --input step.xyz --gradient gradient.json"
            )
            engine_row.addWidget(self.link_engine_command, stretch=1)
            layout.addLayout(engine_row)

            out_row = QHBoxLayout()
            out_row.addWidget(QLabel("Run dir"))
            self.link_run_dir = QLineEdit()
            out_browse = QPushButton("Browse")
            out_browse.clicked.connect(self.browse_link_run_dir)
            self.link_coordinate_model = QComboBox()
            self.link_coordinate_model.addItems(("gic", "cartesian"))
            self.link_active_space = QComboBox()
            self.link_active_space.addItems(("total_symmetric", "all", "cartesian"))
            out_row.addWidget(self.link_run_dir, stretch=1)
            out_row.addWidget(out_browse)
            out_row.addWidget(QLabel("Coordinates"))
            out_row.addWidget(self.link_coordinate_model)
            out_row.addWidget(QLabel("Active"))
            out_row.addWidget(self.link_active_space)
            layout.addLayout(out_row)

            numeric_row = QHBoxLayout()
            self.link_max_steps = QLineEdit("50")
            self.link_trust_radius = QLineEdit("0.2")
            self.link_gradient_tolerance = QLineEdit("1e-5")
            self.link_step_tolerance = QLineEdit("1e-5")
            self.link_energy_tolerance = QLineEdit("1e-8")
            prepare_button = QPushButton("Prepare LINK")
            prepare_button.clicked.connect(self.run_link_prepare)
            numeric_row.addWidget(QLabel("Steps"))
            numeric_row.addWidget(self.link_max_steps)
            numeric_row.addWidget(QLabel("Trust"))
            numeric_row.addWidget(self.link_trust_radius)
            numeric_row.addWidget(QLabel("Grad tol"))
            numeric_row.addWidget(self.link_gradient_tolerance)
            numeric_row.addWidget(QLabel("Step tol"))
            numeric_row.addWidget(self.link_step_tolerance)
            numeric_row.addWidget(QLabel("Energy tol"))
            numeric_row.addWidget(self.link_energy_tolerance)
            numeric_row.addWidget(prepare_button)
            layout.addLayout(numeric_row)

            self.link_summary = QTextEdit()
            self.link_summary.setReadOnly(True)
            self.link_summary.setMaximumHeight(150)
            layout.addWidget(self.link_summary)

            self.link_table_tabs = QTabWidget()
            self.link_settings_table = QTableWidget(0, 2)
            self.link_output_table = QTableWidget(0, 2)
            self.link_table_tabs.addTab(self.link_settings_table, "Settings")
            self.link_table_tabs.addTab(self.link_output_table, "Outputs")
            layout.addWidget(self.link_table_tabs, stretch=1)
            return tab

        def browse_link_run_dir(self) -> None:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select LINK run directory",
                self.link_run_dir.text().strip() or str(Path.cwd()),
            )
            if path:
                self.link_run_dir.setText(path)

        def run_link_prepare(self) -> None:
            if not self._ensure_link_ready_for_command("LINK"):
                return
            engine_command = self.link_engine_command.text().strip()
            if not engine_command:
                QMessageBox.warning(self, "LINK", "Enter an external engine command first.")
                return
            max_steps = self._required_positive_int_field(
                self.link_max_steps,
                "LINK",
                "max steps",
            )
            if max_steps is None:
                return
            trust_radius = self._optional_float_field(
                self.link_trust_radius,
                "LINK",
                "trust radius",
                default=0.2,
            )
            if trust_radius is False:
                return
            gradient_tolerance = self._optional_float_field(
                self.link_gradient_tolerance,
                "LINK",
                "gradient tolerance",
                default=1.0e-5,
            )
            if gradient_tolerance is False:
                return
            step_tolerance = self._optional_float_field(
                self.link_step_tolerance,
                "LINK",
                "step tolerance",
                default=1.0e-5,
            )
            if step_tolerance is False:
                return
            energy_tolerance = self._optional_float_field(
                self.link_energy_tolerance,
                "LINK",
                "energy tolerance",
                default=1.0e-8,
            )
            if energy_tolerance is False:
                return
            self.link_controller.set_xyzin(self.controller.xyzin)
            command = self.link_controller.prepare_command(
                run_dir=self._link_run_dir(),
                engine_command=engine_command,
                coordinate_model=self.link_coordinate_model.currentText(),
                active_space=self.link_active_space.currentText(),
                max_steps=max_steps,
                trust_radius=trust_radius,
                gradient_tolerance=gradient_tolerance,
                step_tolerance=step_tolerance,
                energy_tolerance=energy_tolerance,
            )
            self._start_command(command, command.label)

        def _ensure_link_ready_for_command(self, title: str) -> bool:
            if self.controller.xyzin is None:
                QMessageBox.warning(self, title, "Open or preprocess a MATRIX xyzin first.")
                return False
            if self.process is not None:
                QMessageBox.information(self, "ORACLE", "A command is already running.")
                return False
            self.link_controller.set_xyzin(self.controller.xyzin)
            return True

        def _link_run_dir(self) -> Path:
            if self.controller.xyzin is None:
                raise ValueError("no MATRIX xyzin project is loaded")
            text = self.link_run_dir.text().strip()
            run_dir = Path(text) if text else default_link_run_dir(self.controller.xyzin)
            self.link_run_dir.setText(str(run_dir))
            return run_dir

        def _required_positive_int_field(
            self,
            field: QLineEdit,
            title: str,
            label: str,
        ) -> int | None:
            text = field.text().strip()
            try:
                value = int(text)
            except ValueError:
                QMessageBox.warning(self, title, f"Invalid {label}: {text}")
                return None
            if value <= 0:
                QMessageBox.warning(self, title, f"{label} must be positive.")
                return None
            return value

        def _set_default_link_outputs(self, xyzin: Path) -> None:
            self.link_run_dir.setText(str(default_link_run_dir(xyzin)))

        def _populate_link_tables(self, xyzin: Path) -> None:
            state = load_link_gui_state(xyzin)
            self.link_summary.setPlainText("\n".join(link_gui_state_lines(state)))
            self._fill_table(self.link_settings_table, state.settings)
            self._fill_table(self.link_output_table, state.outputs)

        def _clear_link_tables(self) -> None:
            summary = getattr(self, "link_summary", None)
            if summary is not None:
                summary.clear()
            for table in (
                getattr(self, "link_settings_table", None),
                getattr(self, "link_output_table", None),
            ):
                if table is not None:
                    table.setRowCount(0)

        def _fill_table(
            self,
            widget: QTableWidget,
            table: (
                StructureTable
                | ElectronicTable
                | GICForgeTable
                | GFTable
                | SEFitTable
                | ThermoKineticsTable
                | LinkTable
                | ToolContractTable
                | WorkbenchTable
            ),
        ) -> None:
            widget.setColumnCount(len(table.columns))
            widget.setHorizontalHeaderLabels(table.columns)
            widget.setRowCount(len(table.rows))
            for row_index, row in enumerate(table.rows):
                for column_index, value in enumerate(row):
                    widget.setItem(row_index, column_index, QTableWidgetItem(str(value)))
            widget.resizeColumnsToContents()

    app = QApplication([])
    window = OracleDashboardWindow(initial_xyzin)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
