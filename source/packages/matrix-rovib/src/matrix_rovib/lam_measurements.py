"""One measurement model for resolved transitions and intensity-weighted blends."""

from __future__ import annotations
import numpy as np
import hashlib
from matrix_numerics.sternheimer import DeflatedResponseSolver
from .lam_response import apply_factors, k_contraction_response
from .lam_errors import LAMTruncationError


def response_factors(
    surface,
    solved,
    variables,
    values,
    column,
    manifest,
    parameters,
    response_states,
    *,
    moving_basis=True,
    maximum_rotation_order=None,
    parameter_reference_surface=None,
):
    from .lam_fit import (
        _surface_parameter_direction,
        _lam_direction_action,
        _offset_surface,
        _projected_coupled_factors,
    )
    from .lam_rotation import solve_lam_basis
    from .general_lam_operator import compile_general_lam_factors

    if manifest is not None and maximum_rotation_order is not None:
        from .general_lam_operator import truncate_rotational_order

        manifest = truncate_rotational_order(manifest, maximum_rotation_order)
    variable = variables[column]
    if "_native_surface_response" in variable:
        return variable["_native_surface_response"]
    u = solved.basis.vectors
    du = np.zeros_like(u)
    if variable["path"].startswith("operator."):
        changed = dict(parameters)
        changed[variable["path"][9:]] += variable["multiplier"]
        operators = {}
        for J in solved.blocks:
            a = compile_general_lam_factors(
                surface,
                solved.basis,
                J,
                manifest,
                representation=solved.representation,
                parameters_MHz=changed,
            )
            b = compile_general_lam_factors(
                surface,
                solved.basis,
                J,
                manifest,
                representation=solved.representation,
                parameters_MHz=parameters,
            )
            operators[J] = a + tuple((-left, right) for left, right in b)
        return operators, du
    original=surface if parameter_reference_surface is None else parameter_reference_surface
    direction = _surface_parameter_direction(original, variables, values, column)
    action = _lam_direction_action(surface, direction, u)
    size = solved.basis.primitive_size or len(u)
    count = min(size, u.shape[1] + response_states)
    if moving_basis and np.any(action) and count > u.shape[1]:
        extended = solve_lam_basis(
            surface,
            state_count=count,
            primitive_basis=solved.basis.primitive_representation,
            fourier_cutoffs=solved.basis.fourier_cutoffs,
        )
        q = extended.vectors[:, u.shape[1] :]
        gaps = solved.basis.energies_MHz[None, :] - extended.energies_MHz[u.shape[1] :, None]
        if np.any(abs(gaps) < 1.0e-10):
            raise LAMTruncationError("lam_states", "LAM response cuts a degenerate manifold")
        du = q @ ((q.conj().T @ action) / gaps)
    step = 1.0e-3 / max(1.0, float(np.max(abs(du), initial=0.0)))
    operators = {}
    for J in solved.blocks:
        factors = []
        for mult, coefficient in ((-2.0, 1.0), (-1.0, -8.0), (1.0, 8.0), (2.0, -1.0)):
            local = _projected_coupled_factors(
                _offset_surface(surface, direction, mult * step),
                u + mult * step * du,
                J,
                solved.basis,
                operator_manifest=manifest,
                operator_parameters_MHz=parameters,
                representation=solved.representation,
                maximum_rotation_order=maximum_rotation_order,
            )
            factors.extend((coefficient / (12 * step) * left, right) for left, right in local)
        operators[J] = tuple(factors)
    return operators, du


def _degenerate_indices(block, index):
    level = block.levels[index]
    tolerance = max(1.0e-8, 128 * np.spacing(max(1.0, abs(level.energy_MHz))))
    return [
        k
        for k, l in enumerate(block.levels)
        if l.symmetry_sector == level.symmetry_sector
        and l.rovibrational_irrep == level.rovibrational_irrep
        and abs(l.energy_MHz - level.energy_MHz) <= tolerance
    ]


def _eigen_response(
    solved, J, index, factors, dfactors, moving, *, workspace=None, response_key=None
):
    block = solved.blocks[J]
    level = block.levels[index]
    nrot = 2 * J + 1
    nlam = solved.basis.vectors.shape[1]
    indices = np.flatnonzero(np.asarray(solved.basis.symmetry_sectors) == level.symmetry_sector)
    # A full molecular action may mix translation labels: then use its whole span.
    if not len(indices):
        indices = np.arange(nlam)
    rows = (indices[:, None] * nrot + np.arange(nrot)).reshape(-1)
    local = lambda fs: tuple((a[np.ix_(indices, indices)], b) for a, b in fs)
    lf, ld = local(factors), local(dfactors)
    h = lambda v: apply_factors(lf, v, len(indices), nrot)
    dh = lambda v: apply_factors(ld, v, len(indices), nrot)
    candidates = _degenerate_indices(block, index)
    states = block.vectors[np.ix_(rows, candidates)]
    energy = level.energy_MHz
    contraction = None if solved.k_contractions is None else solved.k_contractions[J]
    embedding = None
    dembedding = None
    if contraction is not None:
        sub = next(s for s in contraction.subspaces if s.sector == level.symmetry_sector)
        embedding = sub.embedding
        if moving:
            full = np.zeros((nlam * nrot, embedding.shape[1]), complex)
            full[rows] = embedding
            dembedding = k_contraction_response(contraction, factors, dfactors, full)[rows]
        else:
            dembedding = np.zeros_like(embedding)
    e = lambda v: v if embedding is None else embedding @ v
    et = lambda v: v if embedding is None else embedding.conj().T @ v
    small = et(states)

    def derivative_on(v):
        z = e(v)
        result = et(dh(z))
        if embedding is not None:
            result += et(h(dembedding @ v)) + dembedding.conj().T @ h(z)
        return result

    split = small.conj().T @ derivative_on(small)
    if len(candidates) > 1 and not np.allclose(
        split, np.trace(split) / len(candidates) * np.eye(len(candidates)), rtol=1.0e-7, atol=1.0e-6
    ):
        raise RuntimeError(
            "parameter splits an unresolved degenerate multiplet; specify a differentiable assignment"
        )
    derivatives = []
    dim = len(small)
    cache = {} if workspace is None else workspace.setdefault("prepared", {})
    signature = hashlib.sha256()
    for array in (block.vectors[rows], block.energies_MHz, *[x for pair in lf for x in pair]):
        array = np.ascontiguousarray(array)
        signature.update(str((array.shape, array.dtype.str)).encode())
        signature.update(array.tobytes())
    if embedding is not None:
        signature.update(np.ascontiguousarray(embedding).tobytes())
    cache_key = (J, index, signature.hexdigest())
    prepared = cache.get(cache_key)
    if isinstance(prepared,dict):
        restored=DeflatedResponseSolver(lambda z:et(h(e(z))),prepared['diagonal'],prepared['energies'],
            prepared['vectors'],prepared['energy'],prepared['cluster'])
        restored.counts.update(prepared['counts']);restored.warm.update(prepared['warm'])
        cache[cache_key]=prepared=restored
    if prepared is None:
        models = {} if workspace is None else workspace.setdefault("models", {})
        model_key = (J, level.symmetry_sector, signature.hexdigest())
        if model_key not in models:
            sector_indices = [
                k for k, l in enumerate(block.levels) if l.symmetry_sector == level.symmetry_sector
            ]
            known = np.asarray(et(block.vectors[np.ix_(rows, sector_indices)]), complex)
            if embedding is None:
                diagonal = sum(
                    (np.kron(np.diag(a), np.diag(b)) for a, b in lf), start=np.zeros(dim, complex)
                ).real
            else:
                diagonal = np.real(np.sum(embedding.conj() * h(embedding), axis=0))
            for key in tuple(models):
                if key[:2] == model_key[:2]:
                    models.pop(key)
            models[model_key] = (sector_indices, known, diagonal)
        sector_indices, known, diagonal = models[model_key]
        prepared = DeflatedResponseSolver(
            lambda z: et(h(e(z))),
            diagonal,
            np.array([block.levels[k].energy_MHz for k in sector_indices]),
            known,
            energy,
            [sector_indices.index(k) for k in candidates],
        )
        for key in tuple(cache):
            if key[:2] == cache_key[:2]:
                cache.pop(key)
        cache[cache_key] = prepared
    elif workspace is not None:
        workspace["cache_hits"] = workspace.get("cache_hits", 0) + 1
    # Reject an incomplete degenerate manifold at a truncated root boundary.
    same_sector = [
        k for k, l in enumerate(block.levels) if l.symmetry_sector == level.symmetry_sector
    ]
    if candidates[-1] == same_sector[-1] and len(same_sector) < dim:
        raise LAMTruncationError(
            "roots", "blend response reaches the root cutoff; increase levels_per_j"
        )
    for k in range(len(candidates)):
        rhs = -derivative_on(small[:, k])
        warm_key = (J, index, response_key, k)
        previous = None if workspace is None else workspace.setdefault("warm", {}).get(warm_key)
        dy = prepared.solve(rhs, warm_start=previous)
        if workspace is not None:
            warm = workspace["warm"]
            previous = warm.pop(warm_key, None)
            workspace["warm_bytes"] = (
                workspace.get("warm_bytes", 0)
                - (0 if previous is None else previous.nbytes)
                + dy.nbytes
            )
            warm[warm_key] = dy.copy()
            limit = workspace.get("memory_budget_bytes", 64 * 1024**2)
            while warm and workspace["warm_bytes"] > limit:
                removed = warm.pop(next(iter(warm)))
                workspace["warm_bytes"] -= removed.nbytes
        physical = e(dy)
        if embedding is not None:
            physical += dembedding @ small[:, k]
        full = np.zeros(nlam * nrot, complex)
        full[rows] = physical
        derivatives.append(full)
    return candidates, np.column_stack(derivatives)


def native_strengths(context, values, components, *, derivatives=False):
    from .lam_fit import _projected_coupled_factors, _lam_level_index
    from .native_rotor import dipole_transition_matrix

    predicted, surface, solved, mapping = context["predict"](values)
    dipole = surface.dipole_au
    if dipole is None:
        vector = np.asarray(context["dipole_au"], float)
        if vector.shape != (3,):
            raise ValueError("native blends require a dipole surface or dipole_au vector")
        dipole = np.broadcast_to(vector, surface.grid_shape + (3,))
    field = np.asarray(dipole).reshape(-1, 3)
    u = solved.basis.vectors
    mu = tuple(u.conj().T @ (field[:, axis, None] * u) for axis in range(3))
    observations = context["observations"]
    nvar = len(values)
    strengths = np.zeros(len(components))
    jac = np.zeros((len(components), nvar))
    pairs = []
    for row, i in enumerate(components):
        obs = observations[i]
        lower, upper = obs["lower"], obs["upper"]
        pairs.append(
            (
                lower.J,
                upper.J,
                _lam_level_index(lower, solved, mapping),
                _lam_level_index(upper, solved, mapping),
            )
        )
    temperature = context.get("temperature_K")
    beta = None
    if temperature is not None:
        from .native_rotor import MHZ_TO_KELVIN

        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("blend temperature must be positive")
        beta = MHZ_TO_KELVIN / temperature
    parameters = solved.operator_parameters_MHz
    factors = (
        {
            J: _projected_coupled_factors(
                surface,
                u,
                J,
                solved.basis,
                operator_manifest=context["operator_manifest"],
                operator_parameters_MHz=parameters,
                representation=solved.representation,
            )
            for J in solved.blocks
        }
        if derivatives
        else {}
    )

    def moment(lvec, uvec, jl, ju, operators):
        result = np.zeros((lvec.shape[1], uvec.shape[1]), complex)
        for axis, name in enumerate(("a", "b", "c")):
            rot = dipole_transition_matrix(jl, ju, name, solved.representation)
            result += np.einsum(
                "ari,ab,rs,bsj->ij",
                lvec.conj().reshape(u.shape[1], 2 * jl + 1, -1),
                operators[axis],
                rot,
                uvec.reshape(u.shape[1], 2 * ju + 1, -1),
                optimize=True,
            )
        return result

    clusters = []
    for jl, ju, il, iu in pairs:
        lo, up = (
            _degenerate_indices(solved.blocks[jl], il),
            _degenerate_indices(solved.blocks[ju], iu),
        )
        clusters.append((lo, up))
    for row, ((jl, ju, il, iu), (lo, up)) in enumerate(zip(pairs, clusters, strict=True)):
        value = moment(
            solved.blocks[jl].vectors[:, lo], solved.blocks[ju].vectors[:, up], jl, ju, mu
        )
        strengths[row] = np.sum(abs(value) ** 2) * solved.blocks[jl].levels[il].spin_weight
    raw_strengths = strengths.copy()
    population = np.ones(len(strengths))
    if beta is not None:
        for row, (jl, ju, il, iu) in enumerate(pairs):
            frequency = predicted[components[row]]
            if frequency <= 0:
                raise ValueError("thermal blend intensities require positive frequencies")
            lower = solved.blocks[jl].levels[il].energy_MHz - solved.ground_energy_MHz
            population[row] = frequency * (-np.expm1(-beta * frequency)) * np.exp(-beta * lower)
        strengths *= population
    if derivatives:
        for column in range(nvar):
            df, du = response_factors(
                surface,
                solved,
                context["variables"],
                values,
                column,
                context["operator_manifest"],
                parameters,
                context["options"].get("basis_response_states", 48),
                moving_basis=context["moving_k_contraction"],
                parameter_reference_surface=context.get("parameter_reference_surface"),
            )
            dmu = tuple(
                du.conj().T @ (field[:, axis, None] * u) + u.conj().T @ (field[:, axis, None] * du)
                for axis in range(3)
            )
            dipole_direction = context["variables"][column].get("dipole_direction")
            if dipole_direction is not None:
                dfield = np.asarray(dipole_direction).reshape(-1, 3)
                dmu = tuple(dmu[a] + u.conj().T @ (dfield[:, a, None] * u) for a in range(3))
            cached = {}
            for row, ((jl, ju, il, iu), (lo, up)) in enumerate(zip(pairs, clusters, strict=True)):
                for J, index in ((jl, il), (ju, iu)):
                    if (J, index) not in cached:
                        cached[J, index] = _eigen_response(
                            solved,
                            J,
                            index,
                            factors[J],
                            df[J],
                            context["moving_k_contraction"],
                            workspace=context.setdefault("response_workspace", {}),
                            response_key=column,
                        )[1]
                lv, uv = solved.blocks[jl].vectors[:, lo], solved.blocks[ju].vectors[:, up]
                amplitude = moment(lv, uv, jl, ju, mu)
                delta = (
                    moment(cached[jl, il], uv, jl, ju, mu)
                    + moment(lv, cached[ju, iu], jl, ju, mu)
                    + moment(lv, uv, jl, ju, dmu)
                )
                spin = solved.blocks[jl].levels[il].spin_weight
                jac[row, column] = (
                    2 * np.real(np.sum(amplitude.conj() * delta)) * spin * population[row]
                )
                if beta is not None:

                    def energy_response(J, index):
                        v = solved.blocks[J].vectors[:, index : index + 1]
                        c = None if solved.k_contractions is None else solved.k_contractions[J]
                        if c is not None and context["moving_k_contraction"]:
                            from .lam_response import moving_ritz_derivatives

                            return moving_ritz_derivatives(
                                c,
                                factors[J],
                                df[J],
                                np.array([solved.blocks[J].levels[index].energy_MHz]),
                                v,
                            )[0]
                        return np.real(
                            np.sum(v.conj() * apply_factors(df[J], v, u.shape[1], 2 * J + 1))
                        )

                    dl = energy_response(jl, il)
                    dnu = energy_response(ju, iu) - dl
                    ground_J, ground_index = min(
                        (
                            (J, k)
                            for J, block in solved.blocks.items()
                            for k in range(len(block.levels))
                        ),
                        key=lambda key: solved.blocks[key[0]].levels[key[1]].energy_MHz,
                    )
                    dl -= energy_response(ground_J, ground_index)
                    frequency = predicted[components[row]]
                    z = beta * frequency
                    stimulated = beta / np.expm1(z) if z < 700 else 0.0
                    jac[row, column] += (
                        raw_strengths[row]
                        * population[row]
                        * ((1 / frequency + stimulated) * dnu - beta * dl)
                    )
    return strengths, jac
