from __future__ import annotations

from pathlib import Path

from .commands import (
    OracleGuiCommand,
    gaussian_fchk_summary_command,
    gaussian_formchk_command,
    gaussian_promote_electronic_command,
    gaussian_promote_fchk_command,
    gaussian_promote_rovib_command,
    gaussian_run_command,
    gaussian_status_command,
    gaussian_summary_command,
    gicforge_gaussian_input_command,
    molpro_molden_command,
    molpro_promote_command,
    molpro_run_command,
    molpro_summary_command,
    molpro_status_command,
    mrcc_promote_command,
    mrcc_summary_command,
    pyscf_run_command,
    pyscf_status_command,
    pyscf_summary_command,
    qm_remote_fetch_command,
    qm_remote_status_command,
    qm_remote_submit_command,
    orca_molden_command,
    orca_promote_command,
    orca_run_command,
    orca_summary_command,
    orca_status_command,
    xtb_run_command,
    xtb_status_command,
    xtb_summary_command,
)


class OracleQMJobsController:
    def __init__(self, xyzin: Path | str | None = None) -> None:
        self.xyzin = None if xyzin is None else Path(xyzin)

    def set_xyzin(self, xyzin: Path | str | None) -> None:
        self.xyzin = None if xyzin is None else Path(xyzin)

    def gaussian_input_command(
        self,
        output: Path | str,
        *,
        route: str,
        title: str | None = None,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return gicforge_gaussian_input_command(self.xyzin, output, route=route, title=title)

    def remote_submit_command(
        self,
        input_path: Path | str,
        *,
        engine: str,
        host: str = "",
        remote_root: str = "",
        extra_args: tuple[str, ...] = (),
    ) -> OracleGuiCommand:
        return qm_remote_submit_command(
            input_path,
            engine=engine,
            host=host,
            remote_root=remote_root,
            extra_args=extra_args,
        )

    def remote_status_command(
        self,
        *,
        host: str = "",
        remote_root: str = "",
    ) -> OracleGuiCommand:
        return qm_remote_status_command(host=host, remote_root=remote_root)

    def remote_fetch_command(
        self,
        job: str,
        *,
        host: str = "",
        remote_root: str = "",
        destination: Path | str = "remote_qm_runs",
        promote: str = "none",
        xyzin: Path | str | None = None,
    ) -> OracleGuiCommand:
        target_xyzin = xyzin if xyzin is not None else self.xyzin
        return qm_remote_fetch_command(
            job,
            host=host,
            remote_root=remote_root,
            destination=destination,
            promote=promote,
            xyzin=target_xyzin,
        )

    def gaussian_status_command(self, workdir: Path | str) -> OracleGuiCommand:
        return gaussian_status_command(workdir)

    def gaussian_run_command(
        self,
        workdir: Path | str,
        *,
        executable: str | None = None,
        input_path: Path | str | None = None,
        background: bool = False,
        timeout: float | None = None,
    ) -> OracleGuiCommand:
        return gaussian_run_command(
            workdir,
            executable=executable,
            input_path=input_path,
            background=background,
            timeout=timeout,
        )

    def gaussian_formchk_command(
        self,
        chk: Path | str,
        fchk: Path | str | None = None,
        *,
        executable: str | None = None,
        timeout: float | None = None,
    ) -> OracleGuiCommand:
        return gaussian_formchk_command(chk, fchk, executable=executable, timeout=timeout)

    def gaussian_fchk_summary_command(self, fchk: Path | str) -> OracleGuiCommand:
        return gaussian_fchk_summary_command(fchk)

    def gaussian_log_summary_command(self, log: Path | str) -> OracleGuiCommand:
        return gaussian_summary_command(log)

    def gaussian_promote_fchk_command(
        self,
        fchk: Path | str,
        *,
        cartesian_hessian: bool = True,
        normal_modes: bool = True,
        qff: bool = True,
        electronic: bool = True,
        orbitals: bool = True,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return gaussian_promote_fchk_command(
            fchk,
            self.xyzin,
            cartesian_hessian=cartesian_hessian,
            normal_modes=normal_modes,
            qff=qff,
            electronic=electronic,
            orbitals=orbitals,
        )

    def gaussian_promote_electronic_command(
        self,
        log: Path | str,
        *,
        electronic: bool = True,
        transitions: bool = True,
        orbital_files: tuple[Path | str, ...] = (),
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return gaussian_promote_electronic_command(
            log,
            self.xyzin,
            electronic=electronic,
            transitions=transitions,
            orbital_files=orbital_files,
        )

    def gaussian_promote_rovib_command(
        self,
        log: Path | str,
        *,
        vibrational: bool = True,
        rotational: bool = True,
        deltabvib: bool = True,
        invert_imaginary: bool = True,
        exclude_modes: tuple[int, ...] = (),
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return gaussian_promote_rovib_command(
            log,
            self.xyzin,
            vibrational=vibrational,
            rotational=rotational,
            deltabvib=deltabvib,
            invert_imaginary=invert_imaginary,
            exclude_modes=exclude_modes,
        )

    def molpro_summary_command(self, output: Path | str) -> OracleGuiCommand:
        return molpro_summary_command(output)

    def molpro_status_command(
        self,
        workdir: Path | str,
        *,
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
    ) -> OracleGuiCommand:
        return molpro_status_command(workdir, input_path=input_path, output_path=output_path)

    def molpro_run_command(
        self,
        workdir: Path | str,
        *,
        executable: str | None = None,
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
        background: bool = False,
        timeout: float | None = None,
        extra_args: tuple[str, ...] = (),
    ) -> OracleGuiCommand:
        return molpro_run_command(
            workdir,
            executable=executable,
            input_path=input_path,
            output_path=output_path,
            background=background,
            timeout=timeout,
            extra_args=extra_args,
        )

    def molpro_promote_command(self, output: Path | str) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return molpro_promote_command(output, self.xyzin)

    def molpro_molden_command(
        self,
        output: Path | str,
        *,
        molden: Path | str | None = None,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return molpro_molden_command(output, self.xyzin, molden=molden)

    def orca_status_command(
        self,
        workdir: Path | str,
        *,
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
    ) -> OracleGuiCommand:
        return orca_status_command(workdir, input_path=input_path, output_path=output_path)

    def orca_run_command(
        self,
        workdir: Path | str,
        *,
        executable: str | None = None,
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
        background: bool = False,
        timeout: float | None = None,
        extra_args: tuple[str, ...] = (),
    ) -> OracleGuiCommand:
        return orca_run_command(
            workdir,
            executable=executable,
            input_path=input_path,
            output_path=output_path,
            background=background,
            timeout=timeout,
            extra_args=extra_args,
        )

    def orca_summary_command(self, output: Path | str) -> OracleGuiCommand:
        return orca_summary_command(output)

    def orca_promote_command(self, output: Path | str) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return orca_promote_command(output, self.xyzin)

    def orca_molden_command(
        self,
        gbw: Path | str,
        *,
        output: Path | str | None = None,
        executable: str | None = None,
        timeout: float | None = None,
    ) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return orca_molden_command(
            gbw,
            self.xyzin,
            output=output,
            executable=executable,
            timeout=timeout,
        )

    def mrcc_summary_command(self, output: Path | str) -> OracleGuiCommand:
        return mrcc_summary_command(output)

    def xtb_summary_command(
        self,
        output: Path | str,
        *,
        geometry: Path | str | None = None,
    ) -> OracleGuiCommand:
        return xtb_summary_command(output, geometry=geometry)

    def xtb_status_command(
        self,
        workdir: Path | str,
        *,
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
    ) -> OracleGuiCommand:
        return xtb_status_command(workdir, input_path=input_path, output_path=output_path)

    def xtb_run_command(
        self,
        workdir: Path | str,
        *,
        executable: str | None = None,
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
        background: bool = False,
        timeout: float | None = None,
        extra_args: tuple[str, ...] = (),
    ) -> OracleGuiCommand:
        return xtb_run_command(
            workdir,
            executable=executable,
            input_path=input_path,
            output_path=output_path,
            background=background,
            timeout=timeout,
            extra_args=extra_args,
        )

    def pyscf_summary_command(self, output: Path | str) -> OracleGuiCommand:
        return pyscf_summary_command(output)

    def pyscf_status_command(
        self,
        workdir: Path | str,
        *,
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
    ) -> OracleGuiCommand:
        return pyscf_status_command(workdir, input_path=input_path, output_path=output_path)

    def pyscf_run_command(
        self,
        workdir: Path | str,
        *,
        executable: str | None = None,
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
        background: bool = False,
        timeout: float | None = None,
        extra_args: tuple[str, ...] = (),
    ) -> OracleGuiCommand:
        return pyscf_run_command(
            workdir,
            executable=executable,
            input_path=input_path,
            output_path=output_path,
            background=background,
            timeout=timeout,
            extra_args=extra_args,
        )

    def mrcc_promote_command(self, output: Path | str) -> OracleGuiCommand:
        if self.xyzin is None:
            raise ValueError("no MATRIX xyzin project is loaded")
        return mrcc_promote_command(output, self.xyzin)


def default_qm_gaussian_input_output(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.gic.gjf")


def default_qm_gaussian_workdir(xyzin: Path | str) -> Path:
    target = Path(xyzin)
    return target.with_name(f"{target.stem}.gaussian")


def default_qm_formchk_output(chk: Path | str) -> Path:
    target = Path(chk)
    return target.with_suffix(".fchk")
