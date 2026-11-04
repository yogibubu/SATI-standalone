"""Config parsing for the fit prototype.

INI sections (minimal):

[system]
# natoms optional (deduced from data)
# nvib optional (deduced from U or nprim)

[basis]
# per-coordinate basis modes and parameters
# default_mode = poly
# mode_i = poly|rational|rational2|trig|trig_cos|legendre|legendre_cos|morse|exp
# shift_i = S (for rational)
# m_i = m (associated Legendre order)
# a_i = Morse/exp parameter
# x0_i = Morse/exp shift

[fit]
# robust = huber
# delta = 1.0
# lambda = 1.0e-6
# max_iter = 50
# tol = 1.0e-10
# fd_step = 1.0e-4
# scale = mad|std|none

[u]
# u_path = path/to/U.npy   (shape nprim x nvib)

[terms]
# term_list = path/to/terms.txt  (optional)
# If absent, use default generation rules.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FitConfig:
    delta: float = 1.0
    ridge: float = 1.0e-6
    max_iter: int = 50
    tol: float = 1.0e-10
    fd_step: float = 1.0e-4


@dataclass
class UConfig:
    symmetry_mode: str = "hybrid"
    prune_mode: str = "svd"
    zeff_tol: float = 0.05
    geometry_match_tol: float = 12.0
    pattern_report_path: str | None = None
    symmetrize_global: bool = False
    keep_a1_only: bool = False
    symmetry_tol: float = 1.0e-3
    symmetry_max_n: int = 10
    a1_tol: float = 1.0e-6
    assign_symmetry_labels: bool = False
    symmetry_quasi_tol: float | None = None
    symmetry_tol_H: float | None = None
    heavy_only_orient: bool = False
    symmetry_center_idx: int | None = None
    ignore_isotopes: bool = False
    symmetry_max_dev_strict: float | None = None
    symmetry_tol_rel: float = 0.0
    symmetry_auto_max_n: bool = False
    symmetry_inertia_tol: float = 1.0e-3
    symmetry_max_radius: float | None = None
    symmetry_enforce_radial: bool = True
    symmetry_profile: bool = False
    symmetry_group_limit: str | None = None
    symmetry_confidence: bool = False


def load_config(path: str | Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def get_fit_config(cfg: configparser.ConfigParser) -> FitConfig:
    section = cfg["fit"] if "fit" in cfg else {}
    return FitConfig(
        delta=float(section.get("delta", 1.0)),
        ridge=float(section.get("lambda", 1.0e-6)),
        max_iter=int(section.get("max_iter", 50)),
        tol=float(section.get("tol", 1.0e-10)),
        fd_step=float(section.get("fd_step", 1.0e-4)),
    )


def get_u_config(cfg: configparser.ConfigParser) -> UConfig:
    section = cfg["u"] if "u" in cfg else {}
    return UConfig(
        symmetry_mode=str(section.get("symmetry_mode", "hybrid")),
        prune_mode=str(section.get("prune_mode", "svd")),
        zeff_tol=float(section.get("zeff_tol", 0.05)),
        geometry_match_tol=float(section.get("geometry_match_tol", 12.0)),
        pattern_report_path=section.get("pattern_report_path", None),
        symmetrize_global=section.get("symmetrize_global", "false").lower() == "true",
        keep_a1_only=section.get("keep_a1_only", "false").lower() == "true",
        symmetry_tol=float(section.get("symmetry_tol", 1.0e-3)),
        symmetry_max_n=int(section.get("symmetry_max_n", 10)),
        a1_tol=float(section.get("a1_tol", 1.0e-6)),
        assign_symmetry_labels=section.get("assign_symmetry_labels", "false").lower() == "true",
        symmetry_quasi_tol=float(section.get("symmetry_quasi_tol", 0.0)) or None,
        symmetry_tol_H=float(section.get("symmetry_tol_H", 0.0)) or None,
        heavy_only_orient=section.get("heavy_only_orient", "false").lower() == "true",
        symmetry_center_idx=int(section.get("symmetry_center_idx", -1))
        if section.get("symmetry_center_idx", "-1") != "-1"
        else None,
        ignore_isotopes=section.get("ignore_isotopes", "false").lower() == "true",
        symmetry_max_dev_strict=float(section.get("symmetry_max_dev_strict", 0.0)) or None,
        symmetry_tol_rel=float(section.get("symmetry_tol_rel", 0.0)),
        symmetry_auto_max_n=section.get("symmetry_auto_max_n", "false").lower() == "true",
        symmetry_inertia_tol=float(section.get("symmetry_inertia_tol", 1.0e-3)),
        symmetry_max_radius=float(section.get("symmetry_max_radius", 0.0)) or None,
        symmetry_enforce_radial=section.get("symmetry_enforce_radial", "true").lower() == "true",
        symmetry_profile=section.get("symmetry_profile", "false").lower() == "true",
        symmetry_group_limit=section.get("symmetry_group_limit", None),
        symmetry_confidence=section.get("symmetry_confidence", "false").lower() == "true",
    )
