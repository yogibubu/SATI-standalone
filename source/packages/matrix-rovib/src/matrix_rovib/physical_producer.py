"""Reusable SMITH charts and QC/TRINITY potential adapters for physical fits."""

from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import CubicSpline
from .structure_spectrum import cartesian_lam_surface
from .tunneling_coordinates import ROTATIONAL_FACTOR_MHZ_AMU_ANGSTROM2


def grid_derivative(array, coordinates, periodic, axis):
    grid = np.asarray(coordinates[axis])
    a = np.asarray(array)
    if periodic[axis]:
        waves = 2 * np.pi * np.fft.fftfreq(len(grid), grid[1] - grid[0])
        view = [1] * a.ndim
        view[axis] = len(grid)
        result = np.fft.ifft(np.fft.fft(a, axis=axis) * (1j * waves).reshape(view), axis=axis)
        return result.real if not np.iscomplexobj(a) else result
    return CubicSpline(grid, a, axis=axis).derivative()(grid)


def cartesian_lam_surface_response(
    coordinates,
    geometries_angstrom,
    tangents_angstrom,
    masses_amu,
    potential_MHz,
    geometry_directions,
    tangent_directions,
    potential_directions,
    *,
    periodic,
    **surface_fields,
):
    """Exact Cartesian/mass-metric/volume-scalar chain rule on a fixed grid.

    Parameter directions have a leading parameter axis. Masses and the fixed
    coordinate domain do not vary with a structural fitting parameter.
    """
    surface = cartesian_lam_surface(
        coordinates,
        geometries_angstrom,
        tangents_angstrom,
        masses_amu,
        potential_MHz,
        periodic=periodic,
        **surface_fields,
    )
    x = np.array(geometries_angstrom, float, copy=True)
    t = np.array(tangents_angstrom, float, copy=True)
    m = np.asarray(masses_amu, float)
    dx = np.array(geometry_directions, float, copy=True)
    dt = np.array(tangent_directions, float, copy=True)
    dv = np.asarray(potential_directions, float)
    count = len(dx)
    shape = surface.grid_shape
    d = surface.dimension
    if (
        dx.shape != (count,) + x.shape
        or dt.shape != (count,) + t.shape
        or dv.shape != (count,) + shape
        or not all(np.all(np.isfinite(a)) for a in (dx, dt, dv))
    ):
        raise ValueError("inconsistent Cartesian parameter directions")
    x -= np.einsum("a,...ac->...c", m, x)[..., None, :] / m.sum()
    t -= np.einsum("a,...acd->...cd", m, t)[..., None, :, :] / m.sum()
    dx -= np.einsum("a,...ac->...c", m, dx)[..., None, :] / m.sum()
    dt -= np.einsum("a,...acd->...cd", m, dt)[..., None, :, :] / m.sum()
    columns = np.concatenate((t, np.stack([np.cross(a, x) for a in np.eye(3)], axis=-1)), axis=-1)
    metric = np.einsum("...aci,a,...acj->...ij", columns, m, columns, optimize=True)
    inverse = np.linalg.inv(metric)
    invq = np.linalg.inv(inverse[..., :d, :d])
    f = surface.F_MHz
    rho = surface.rho
    factor = ROTATIONAL_FACTOR_MHZ_AMU_ANGSTROM2
    logw = 0.5 * np.linalg.slogdet(metric)[1]
    deriv = lambda a, i: grid_derivative(a, coordinates, periodic, i)
    grad = np.stack([deriv(logw, i) for i in range(d)], axis=-1)
    flux = np.einsum("...ij,...j->...i", f, grad)
    result = []
    for k in range(count):
        dc = np.concatenate(
            (dt[k], np.stack([np.cross(a, dx[k]) for a in np.eye(3)], axis=-1)), axis=-1
        )
        dg = np.einsum("...aci,a,...acj->...ij", dc, m, columns, optimize=True)
        dg += np.swapaxes(dg, -1, -2)
        db = -inverse @ dg @ inverse
        df = factor * db[..., :d, :d]
        drho = -db[..., d:, :d] @ invq - rho @ db[..., :d, :d] @ invq
        dr = (
            factor * db[..., d:, d:]
            - drho @ f @ np.swapaxes(rho, -1, -2)
            - rho @ df @ np.swapaxes(rho, -1, -2)
            - rho @ f @ np.swapaxes(drho, -1, -2)
        )
        dlogw = 0.5 * np.einsum("...ij,...ji->...", inverse, dg)
        dgrad = np.stack([deriv(dlogw, i) for i in range(d)], axis=-1)
        dflux = np.einsum("...ij,...j->...i", df, grad) + np.einsum("...ij,...j->...i", f, dgrad)
        scalar = 0.5 * sum(deriv(dflux[..., i], i) for i in range(d)) + 0.25 * (
            np.einsum("...i,...i->...", dgrad, flux) + np.einsum("...i,...i->...", grad, dflux)
        )
        result.append(
            dict(potential_MHz=dv[k], F_MHz=df, rho=drho, R_MHz=dr, geometric_potential_MHz=scalar)
        )
    return surface, tuple(result)


def smith_chart_response(
    chart, displacements, gic_tangents, displacement_directions, tangent_directions
):
    """Implicit GIC/Eckart first and mixed derivatives using resident sparse AD2."""
    from matrix_smith.evaluation import build_gic_b_matrix
    from matrix_zaff.bmatrix_derivative import build_sparse_gic_b_matrix_derivative

    shifts = np.asarray(displacements, float)
    tg = np.asarray(gic_tangents, float)
    ds = np.asarray(displacement_directions, float)
    dts = np.asarray(tangent_directions, float)
    shape = shifts.shape[:-1]
    rank = chart.definition.rank
    count = len(ds)
    d = tg.shape[-1]
    n = len(chart.masses_amu)
    if (
        shifts.shape != shape + (rank,)
        or tg.shape != shape + (rank, d)
        or ds.shape != (count,) + shifts.shape
        or dts.shape != (count,) + tg.shape
    ):
        raise ValueError("inconsistent GIC chart parameter directions")
    xyz = np.empty(shape + (n, 3))
    t = np.empty(shape + (n, 3, d))
    dx = np.empty((count,) + xyz.shape)
    dt = np.empty((count,) + t.shape)
    fallback = set()
    for at in np.ndindex(shape):
        position = chart.geometry_at_displacement_angstrom(shifts[at])
        xyz[at] = position
        b = np.asarray(build_gic_b_matrix(chart.definition, coordinates_angstrom=position).rows)
        a = np.vstack((b, chart.rigid_constraint_rows))
        tangent = np.linalg.solve(a, np.vstack((tg[at], np.zeros((6, d)))))
        t[at] = tangent.reshape(n, 3, d)
        directions = np.linalg.solve(
            a, np.vstack((ds[(slice(None),) + at].T, np.zeros((6, count))))
        )
        dx[(slice(None),) + at] = directions.T.reshape(count, n, 3)
        prime = build_sparse_gic_b_matrix_derivative(
            chart.definition,
            coordinates_angstrom=position,
            zero_tolerance_per_angstrom2=0.0,
            parallel_workers=1,
        )
        fallback.update(prime.numerical_fallback_primitives)
        if fallback:
            raise NotImplementedError(
                "analytic structural derivatives unavailable for GIC primitives: "
                + ", ".join(sorted(fallback))
            )
        db = np.zeros((count, rank, position.size))
        for section in prime.slices:
            for row_index, row in section.rows:
                for col, value in row.entries:
                    db[:, row_index, col] += directions[section.derivative_column] * value
        rhs = dts[(slice(None),) + at] - np.einsum("pij,jk->pik", db, tangent)
        mixed = np.linalg.solve(
            a,
            np.concatenate((rhs, np.zeros((count, 6, d))), axis=1)
            .transpose(1, 0, 2)
            .reshape(position.size, -1),
        )
        dt[(slice(None),) + at] = (
            mixed.reshape(position.size, count, d).transpose(1, 0, 2).reshape(count, n, 3, d)
        )
    return xyz, t, dx, dt


@dataclass(frozen=True)
class PhysicalSurfaceEvaluation:
    surfaces: dict
    directions: dict
    provenance: dict


class PhysicalLAMProducer:
    """A data-configured, cached SMITH/QC producer; no per-molecule callback.

    Charts and potentials are configured by system. A chart may be a resident
    object or a serialized SMITH GIC definition. The selected GIC displacement
    map is physical input, not a molecular-name dispatch. Potentials support
    Fourier/polynomial terms or a TRINITY dimensionless QFF with its documented
    Cartesian normal-coordinate displacement map.
    """

    def __init__(self, specification):
        if specification.get("schema") != "matrix.lock.physical_producer.v1":
            raise ValueError("unsupported physical producer schema")
        self.specification = deepcopy(specification)
        self.systems = {}
        self.cache = OrderedDict()
        for record in self.specification["systems"]:
            name = str(record["name"])
            if name in self.systems:
                raise ValueError("duplicate physical producer system")
            chart = record.get("chart")
            if chart is None:
                from matrix_smith.gic_serialization import gic_definition_from_dict
                from matrix_smith.eckart_chart import build_eckart_constrained_gic_chart

                raw = record["gic_definition"]
                if isinstance(raw, (str, Path)):
                    raw = json.loads(Path(raw).read_text())
                definition = gic_definition_from_dict(raw)
                chart = build_eckart_constrained_gic_chart(
                    definition,
                    np.asarray(definition.reference_coordinates_angstrom),
                    np.asarray(
                        record.get(
                            "chart_masses_amu",
                            record["potential"].get("masses_amu", record["masses_amu"]),
                        )
                    ),
                    tolerance=float(record.get("chart_tolerance", 1.0e-12)),
                )
            if record["potential"]["representation"] == "trinity_qff":
                potential = record["potential"]
                # Normal-coordinate/QFF masses define a reference frame, not
                # the kinetic isotope. Preserve them across isotope changes.
                masses = np.asarray(potential.get("masses_amu", chart.masses_amu), float)
                potential["masses_amu"] = masses.tolist()
                if not np.array_equal(masses, chart.masses_amu):
                    from matrix_smith.eckart_chart import build_eckart_constrained_gic_chart

                    chart = build_eckart_constrained_gic_chart(
                        chart.definition,
                        chart.reference_angstrom,
                        masses,
                        tolerance=chart.tolerance,
                        max_iterations=chart.max_iterations,
                    )
                if "chart_to_qff_rotation" not in potential:
                    before = np.asarray(chart.reference_angstrom)
                    after = np.asarray(potential["reference_angstrom"])
                    center_before = np.average(before, axis=0, weights=masses)
                    center_after = np.average(after, axis=0, weights=masses)
                    u, _, vh = np.linalg.svd(
                        (before - center_before).T @ (masses[:, None] * (after - center_after))
                    )
                    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vh)]) @ vh
                    potential["chart_to_qff_rotation"] = rotation.tolist()
                    potential["chart_to_qff_translation_angstrom"] = (
                        center_after - center_before @ rotation
                    ).tolist()
            labels = {g.identifier: i for i, g in enumerate(chart.definition.gics)}
            for g in chart.definition.gics:
                matches = [i for i, item in enumerate(chart.definition.gics) if item.name == g.name]
                if len(matches) == 1 and g.name not in labels:
                    labels[g.name] = matches[0]
            if "lam_gic_identifiers" in record:
                selected = record["lam_gic_identifiers"]
                if len(selected) != len(record["coordinates"]) or any(
                    label not in labels for label in selected
                ):
                    raise ValueError("LAM selection must identify one SMITH GIC per coordinate")
                matrix = np.eye(chart.definition.rank)[:, [labels[label] for label in selected]]
                if "lam_gic_map" in record and not np.allclose(matrix, record["lam_gic_map"]):
                    raise ValueError("symbolic LAM selection and explicit GIC map differ")
                record["lam_gic_map"] = matrix.tolist()
            for parameter, offsets in record.get("structure_gic_offsets", {}).items():
                offsets = {offsets: 1.0} if isinstance(offsets, str) else offsets
                vector = np.zeros(chart.definition.rank)
                for label, factor in offsets.items():
                    if label not in labels:
                        raise ValueError("unknown SMITH structural coordinate")
                    vector[labels[label]] += float(factor)
                maps = record.setdefault("structure_gic_maps", {})
                if parameter in maps and not np.allclose(vector, maps[parameter]):
                    raise ValueError("symbolic and explicit structural maps differ")
                maps[parameter] = vector.tolist()
            self.systems[name] = (record, chart)
        if not self.systems:
            raise ValueError("physical producer requires at least one system")
        from matrix_smith.derivative_capabilities import audit_b_prime_capabilities

        self.derivative_capabilities = {
            name: [
                item for item in audit_b_prime_capabilities(chart.definition) if not item.analytic
            ]
            for name, (_, chart) in self.systems.items()
        }
        self.analytic_derivatives_available = not any(self.derivative_capabilities.values())

    def to_specification(self):
        from matrix_smith.gic_serialization import gic_definition_to_dict

        specification = deepcopy(self.specification)
        for record in specification["systems"]:
            chart = self.systems[str(record["name"])][1]
            record.pop("chart", None)
            record["gic_definition"] = gic_definition_to_dict(chart.definition)
            record["masses_amu"] = np.asarray(record.get("masses_amu", chart.masses_amu)).tolist()
            record["chart_tolerance"] = chart.tolerance
            record["chart_masses_amu"] = chart.masses_amu.tolist()
        return json.loads(
            json.dumps(specification, default=lambda a: np.asarray(a).tolist(), allow_nan=False)
        )

    def with_grids(self, system, coordinates):
        """Rebuild chart and metric on a refined grid of the same QC model."""
        from scipy.signal import resample

        specification = self.to_specification()
        for record in specification["systems"]:
            if str(record["name"]) != str(system):
                continue
            old = tuple(np.asarray(q) for q in record["coordinates"])
            new = tuple(np.asarray(q) for q in coordinates)

            def interpolate(value):
                a = np.asarray(value)
                if a.shape[: len(old)] != tuple(len(q) for q in old):
                    return value
                for axis, (before, after, periodic) in enumerate(
                    zip(old, new, record["periodic"], strict=True)
                ):
                    a = (
                        resample(a, len(after), axis=axis)
                        if periodic
                        else CubicSpline(before, a, axis=axis)(after)
                    )
                return a.real.tolist() if not np.iscomplexobj(value) else a.tolist()

            if record["potential"]["representation"] == "grid":
                record["potential"]["values_MHz"] = interpolate(record["potential"]["values_MHz"])
            record["surface_fields"] = {
                k: interpolate(v) for k, v in record.get("surface_fields", {}).items()
            }
            record["coordinates"] = [q.tolist() for q in new]
        return PhysicalLAMProducer(specification)

    def qc_geometry(self, system, coordinate, parameters):
        """One physical point/tangent, for GDV job generation and gradients."""
        from matrix_smith.evaluation import build_gic_b_matrix

        record, chart = self.systems[str(system)]
        lam = np.asarray(record["lam_gic_map"], float)
        q = np.asarray(coordinate, float)
        shift = lam @ q + np.asarray(record.get("gic_offset", np.zeros(chart.definition.rank)))
        for name, vector in record.get("structure_gic_maps", {}).items():
            shift += parameters[name] * np.asarray(vector)
        xyz = chart.geometry_at_displacement_angstrom(shift)
        b = np.asarray(build_gic_b_matrix(chart.definition, coordinates_angstrom=xyz).rows)
        tangent = np.linalg.solve(
            np.vstack((b, chart.rigid_constraint_rows)), np.vstack((lam, np.zeros((6, len(q)))))
        ).reshape(len(xyz), 3, len(q))
        return dict(geometry_angstrom=xyz.tolist(), cartesian_tangents_angstrom=tangent.tolist())

    def __call__(self, parameters):
        return self.evaluate(parameters).surfaces

    def derivatives(self, parameters):
        return self.evaluate(parameters, derivatives=True).directions

    def evaluate(self, parameters, *, derivatives=False):
        names = tuple(parameters)
        key = (names, tuple(float(parameters[n]) for n in names))
        if key in self.cache and (not derivatives or self.cache[key].directions):
            return self.cache[key]
        surfaces = {}
        directions = {}
        provenance = {}
        for name, (record, chart) in self.systems.items():
            grids = tuple(np.asarray(q, float) for q in record["coordinates"])
            periodic = tuple(record["periodic"])
            shape = tuple(len(q) for q in grids)
            d = len(grids)
            rank = chart.definition.rank
            mesh = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1)
            lam = np.asarray(record["lam_gic_map"], float)
            if lam.shape != (rank, d):
                raise ValueError("LAM GIC map must have shape (GIC rank, dimension)")
            shifts = np.einsum("...d,rd->...r", mesh, lam) + np.asarray(
                record.get("gic_offset", np.zeros(rank))
            )
            tangents = np.broadcast_to(lam, shape + lam.shape)
            maps = record.get("structure_gic_maps", {})
            ds = np.zeros((len(names),) + shifts.shape)
            dtg = np.zeros((len(names),) + tangents.shape)
            for k, param in enumerate(names):
                if param in maps:
                    vector = np.asarray(maps[param], float)
                    if vector.shape != (rank,):
                        raise ValueError("structural GIC map must span the chart")
                    shifts += parameters[param] * vector
                    ds[k] = vector
            if derivatives:
                xyz, t, dx, dt = smith_chart_response(chart, shifts, tangents, ds, dtg)
            else:
                from matrix_smith.evaluation import build_gic_b_matrix

                xyz = np.empty(shape + (len(chart.masses_amu), 3))
                t = np.empty(xyz.shape + (d,))
                for at in np.ndindex(shape):
                    xyz[at] = chart.geometry_at_displacement_angstrom(shifts[at])
                    b = np.asarray(
                        build_gic_b_matrix(chart.definition, coordinates_angstrom=xyz[at]).rows
                    )
                    t[at] = np.linalg.solve(
                        np.vstack((b, chart.rigid_constraint_rows)),
                        np.vstack((lam, np.zeros((6, d)))),
                    ).reshape(len(chart.masses_amu), 3, d)
            value, pd = evaluate_physical_potential(
                record["potential"],
                mesh,
                xyz,
                parameters,
                names,
                geometry_directions=dx if derivatives else None,
            )
            fields = record.get("surface_fields", {})
            masses = np.asarray(record.get("masses_amu", chart.masses_amu))
            if derivatives:
                surface, local = cartesian_lam_surface_response(
                    grids, xyz, t, masses, value, dx, dt, pd, periodic=periodic, **fields
                )
                directions[name] = dict(zip(names, local, strict=True))
            else:
                surface = cartesian_lam_surface(
                    grids, xyz, t, masses, value, periodic=periodic, **fields
                )
            surfaces[name] = surface
            provenance[name] = dict(
                chart="SMITH GIC/Eckart",
                potential=record["potential"]["representation"],
                derivative="sparse analytic B-prime and exact mass-metric chain rule"
                if derivatives
                else None,
            )
        result = PhysicalSurfaceEvaluation(surfaces, directions, provenance)
        self.cache[key] = result
        while len(self.cache) > 2:
            self.cache.popitem(last=False)
        return result


def evaluate_physical_potential(
    spec, mesh, geometries, parameters, names, *, geometry_directions=None
):
    """Evaluate declared physical Fourier/polynomial terms or a TRINITY QFF."""
    representation = spec["representation"]
    shape = mesh.shape[:-1]
    result = np.full(shape, float(spec.get("constant_MHz", 0.0)))
    directions = np.zeros((len(names),) + shape)
    if representation == "grid":
        grid = np.asarray(spec["values_MHz"], float)
        if grid.shape != shape:
            raise ValueError("QC potential grid shape mismatch")
        result += grid
    elif representation == "trinity_qff":
        q, dq = qff_normal_coordinates(spec, geometries, geometry_directions)
        # QFF records supply unique derivative indices, with repeated-index factorials.
        from math import factorial

        frequencies = np.asarray(spec["frequencies_cm"], float)
        terms = [
            dict(
                powers=[2 if i == k else 0 for i in range(len(frequencies))],
                coefficient_MHz=0.5 * w * 29979.2458,
            )
            for k, w in enumerate(frequencies)
        ]
        for term in spec.get("derivatives_cm", []):
            indices = term["indices"]
            powers = [indices.count(k) for k in range(len(frequencies))]
            terms.append(
                dict(
                    powers=powers,
                    coefficient_MHz=float(term["value_cm"])
                    * 29979.2458
                    / np.prod([factorial(k) for k in powers]),
                )
            )
        for term in terms:
            powers = term["powers"]
            v = np.ones(shape)
            gradient = np.zeros(shape + (len(powers),))
            for axis, power in enumerate(powers):
                v *= q[..., axis] ** power
            for axis, power in enumerate(powers):
                if power:
                    grad = np.full(shape, float(power))
                    for j, p in enumerate(powers):
                        grad *= q[..., j] ** (p - int(axis == j))
                    gradient[..., axis] = grad
            result += term["coefficient_MHz"] * v
            if dq is not None:
                directions += term["coefficient_MHz"] * np.einsum("...i,p...i->p...", gradient, dq)
    elif representation != "terms":
        raise ValueError("unsupported physical potential representation")
    # Corrections share the same physically identified PES coefficient registry.
    for term in spec.get("terms", []):
        if len(term["factors"]) != mesh.shape[-1]:
            raise ValueError("each potential term must span every LAM coordinate")
        field = np.ones(shape)
        for axis, factor in enumerate(term["factors"]):
            q = mesh[..., axis]
            kind = factor["function"]
            order = int(factor.get("order", 0))
            if order < 0 or order != factor.get("order", 0):
                raise ValueError("PES factors require nonnegative integer orders")
            if kind == "power":
                field *= q**order
            elif kind == "cos":
                field *= np.cos(order * q)
            elif kind == "sin":
                field *= np.sin(order * q)
            elif kind == "one_minus_cos":
                field *= 1 - np.cos(order * q)
            elif kind != "constant":
                raise ValueError("unsupported PES coordinate factor")
        coefficient = float(term.get("coefficient_MHz", 0.0))
        parameter = term.get("parameter")
        multiplier = float(term.get("multiplier_MHz", 1.0))
        if parameter is not None:
            if parameter not in parameters:
                raise ValueError("unknown PES parameter")
            coefficient += multiplier * parameters[parameter]
            directions[names.index(parameter)] += multiplier * field
        result += coefficient * field
    if not np.all(np.isfinite(result)) or not np.all(np.isfinite(directions)):
        raise ValueError("nonfinite physical PES")
    return result, directions


def build_physical_lam_producer(specification, *, base_directory=None):
    base = None if base_directory is None else Path(base_directory)
    if isinstance(specification, (str, Path)):
        source = Path(specification)
        if base is not None and not source.is_absolute():
            source = base / source
        source = source.resolve()
        specification = json.loads(source.read_text())
        base = source.parent
    specification = deepcopy(specification)
    for system in specification["systems"]:
        if isinstance(system.get("gic_definition"), str):
            path = Path(system["gic_definition"])
            if base is not None and not path.is_absolute():
                system["gic_definition"] = str(base / path)
        potential = system["potential"]
        if potential.get("representation") == "trinity_qff_artifact":
            raw = potential["artifact"]
            if isinstance(raw, (str, Path)):
                path = Path(raw)
                if not path.is_absolute() and base is not None:
                    path = base / path
                raw = json.loads(path.read_text())
            system["potential"] = trinity_qff_potential(
                raw, potential["reference_angstrom"], masses_amu=potential.get("masses_amu")
            )
            system["potential"]["terms"] = potential.get("terms", [])
    return PhysicalLAMProducer(specification)


def trinity_qff_potential(fit, reference_angstrom, *, masses_amu=None):
    """Adapt a resident TRINITY QFF result or its JSON without changing units."""
    raw = fit.to_dict() if hasattr(fit, "to_dict") else deepcopy(fit)
    if raw.get("normal_coordinate_convention") != "DIMENSIONLESS_HARMONIC_OSCILLATOR":
        raise ValueError("TRINITY QFF requires its dimensionless-normal-coordinate convention")
    record = dict(
        representation="trinity_qff",
        energy_kind="electronic",
        reference_angstrom=np.asarray(reference_angstrom, float).tolist(),
        displacement_bohr_per_dimensionless=raw["cartesian_displacement_bohr_per_dimensionless"],
        frequencies_cm=raw["harmonic_frequencies_cm-1"],
        derivatives_cm=[
            dict(indices=t["indices_zero_based"], value_cm=t["value_cm-1"])
            for t in raw.get("cubic_terms", []) + raw.get("semidiagonal_quartic_terms", [])
        ],
        source_schema=raw.get("schema"),
        terms=[],
    )

    reference_masses = raw.get("masses_amu") if masses_amu is None else masses_amu
    if reference_masses is not None:
        record["masses_amu"] = np.asarray(reference_masses, float).tolist()
    return record


def qff_normal_coordinates(spec, geometries, geometry_directions=None):
    reference = np.asarray(spec["reference_angstrom"], float)
    displacement = np.asarray(spec["displacement_bohr_per_dimensionless"], float) * 0.529177210903
    masses = np.asarray(spec.get("masses_amu", np.ones(len(reference))), float)
    if (
        displacement.ndim != 2
        or displacement.shape[0] != reference.size
        or masses.shape != (len(reference),)
        or np.any(masses <= 0)
    ):
        raise ValueError("invalid TRINITY normal-coordinate displacement/mass contract")
    weight = np.repeat(masses, 3)
    projection = np.linalg.solve(
        displacement.T @ (weight[:, None] * displacement), displacement.T * weight
    )
    rotation = np.asarray(spec.get("chart_to_qff_rotation", np.eye(3)), float)
    translation = np.asarray(spec.get("chart_to_qff_translation_angstrom", np.zeros(3)), float)
    xyz = np.asarray(geometries) @ rotation + translation
    shape = xyz.shape[:-2]
    q = np.einsum("ij,...j->...i", projection, (xyz - reference).reshape(shape + (-1,)))
    dq = None
    if geometry_directions is not None:
        directions = np.asarray(geometry_directions) @ rotation
        dq = np.einsum(
            "ij,p...j->p...i", projection, directions.reshape((len(directions),) + shape + (-1,))
        )
    return q, dq
