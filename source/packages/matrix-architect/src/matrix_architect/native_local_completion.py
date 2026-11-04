"""Internal quartic fit, protected completion, then a field-only minima audit.

Sparse INTERNAL force constants are closed at zero before fitting. This closure
is a model convention, not an electronic measurement. Normal-mode tensors are
downstream artifacts: their four-distinct entries must not be zeroed afterwards.
No function in this module acquires electronic points or changes QM keywords.
"""
from collections import Counter
from math import factorial, inf

import numpy as np
from scipy.optimize import minimize

from matrix_chem import BOHR_TO_ANGSTROM
from matrix_chem.primitive_coordinates import eval_primitives
from .asymptotic_pic import AsymptoticPicCorrection, AsymptoticPicFactor
from .native_pic_field import NativePicField


def reconstruct_internal_from_local_modal_source(parent, reference, displacements,
                                                 observed_cubic, observed_quartic, *,
                                                 baseline_cubic, baseline_quartic,
                                                 cubic_tolerance_cm=5., quartic_tolerance_cm=5.,
                                                 compact_support=True,
                                                 coefficient_factor_scales=None,
                                                 maximum_scaled_coefficient_hartree=None,
                                                 maximum_solver_iterations=1000,
                                                 additional_native_active_primitives=3,
                                                 equilibrate_source_columns=False,
                                                 minimum_native_energy_norm=False,
                                                 enforce_planar_reflection=False,
                                                 topological_catalog_only=False,
                                                 relative_rank_cutoff=1e-8,
                                                 complete_internal_support=False,
                                                 quartic_ridge_strengths=None,
                                                 quartic_coefficient_energy_limit_hartree=None,
                                                 joint_native_energy_fit=False):
    """Infer native internal parameters from an existing modal source table.

    This is the inverse preparation of local source data, not the final modal
    export. Only native cubic and repeated-index internal quartic products are
    parameters. The independent primitive view is fixed from the geometry,
    and every other correction coefficient starts at zero. No missing modal
    entry is made into an observation; induced modal quartics remain free.
    Parent baselines include frozen nonbonded so it is not counted twice.
    """
    from itertools import combinations_with_replacement
    from scipy.linalg import qr
    from .native_modal_derivatives import native_factor_modal_two_jet, fit_native_modal_derivatives
    from .native_refinement import native_refinement_catalog
    first, second = native_factor_modal_two_jet(parent.primitives,reference,displacements)
    _,r,pivots = qr(first.T,mode='economic',pivoting=True)
    diagonal = abs(np.diag(r))
    if not len(diagonal) or np.count_nonzero(diagonal > diagonal[0]*1e-9) != len(displacements):
        raise ValueError('internal primitive view does not span the local vibrational source')
    selected = tuple(sorted(int(i) for i in pivots[:len(displacements)]))
    factors = [] if compact_support or complete_internal_support else list(native_refinement_catalog(parent.primitives,
        maximum_active_primitives=additional_native_active_primitives))
    if topological_catalog_only and (compact_support or complete_internal_support):
        raise ValueError('a topological native catalog requires noncompact support')
    parameter_indices=range(len(parent.primitives)) if complete_internal_support else selected
    for degree in (() if topological_catalog_only else (3,4)):
        for row in combinations_with_replacement(parameter_indices,degree):
            if degree == 4 and len(set(row)) == 4:
                continue
            factors.append(tuple(AsymptoticPicFactor(i,
                'inverse_distance' if parent.primitives[i].kind == 'bond' else
                'native' if parent.primitives[i].kind == 'angle' else 'sin',power)
                for i,power in sorted(Counter(row).items())))
    factors=tuple(dict.fromkeys(factors))
    if enforce_planar_reflection:
        xyz=np.asarray(reference,float)
        if np.linalg.svd(xyz-xyz.mean(axis=0),compute_uv=False)[-1]>1e-8:
            raise ValueError('planar reflection requires a planar reference')
        q0=np.asarray(parent.field.equilibrium_values)
        odd={i for i,p in enumerate(parent.primitives) if p.kind in ('dihedral','out_of_plane')}
        if any(abs(np.sin(q0[i]))>1e-7 for i in odd):
            raise ValueError('periodic primitives are not at a planar reflection reference')
        factors=tuple(f for f in factors if sum(t.power for t in f if t.primitive_index in odd)%2==0)
    print('local native source catalog',len(factors),'terms; no four-distinct internal quartics',flush=True)
    corrections, report = fit_native_modal_derivatives(first,second,factors,
        observed_cubic,observed_quartic,baseline_cubic,baseline_quartic,
        equilibrate_columns=equilibrate_source_columns,relative_rank_cutoff=relative_rank_cutoff,
        coefficient_factor_scales=coefficient_factor_scales,
        maximum_scaled_coefficient_hartree=maximum_scaled_coefficient_hartree,
        maximum_solver_iterations=maximum_solver_iterations,
        minimum_native_energy_norm=minimum_native_energy_norm,
        quartic_ridge_strengths=quartic_ridge_strengths,
        quartic_coefficient_energy_limit_hartree=quartic_coefficient_energy_limit_hartree,
        joint_native_energy_fit=joint_native_energy_fit)
    passed = (report['3']['maximum_residual_cm'] <= cubic_tolerance_cm
              and report['4']['maximum_residual_cm'] <= quartic_tolerance_cm)
    if maximum_scaled_coefficient_hartree is not None:
        passed = passed and all(report[str(d)]['solver']['success'] for d in (3,4))
    report.update(schema='matrix.architect.internal_source_reconstruction.v1',
                  local_derivatives_passed=passed, selected_internal_indices=list(selected),
                  independent_four_distinct_internal_quartic_parameters=0,
                  missing_internal_correction_parameters_initially_zero=True,
                  compact_support=compact_support,
                  additional_native_active_primitives=0 if compact_support else additional_native_active_primitives,
                  native_corrections_planar_reflection_invariant=bool(enforce_planar_reflection),
                  topological_catalog_only=topological_catalog_only,
                  complete_internal_support=complete_internal_support,
                  relative_rank_cutoff=relative_rank_cutoff,
                  approximate_local_fit_tolerances_cm=[cubic_tolerance_cm,quartic_tolerance_cm],
                  local_source_only=True, external_points_used=False,
                  final_normal_mode_transformation_performed=False,
                  nonbonded_included_in_subtracted_parent_baseline=True)
    field = NativePicField(parent.primitives,parent.field,parent.nonbonded_evaluator,
                          corrections=parent.corrections+corrections,
                          energy_offset_hartree=parent.energy_offset_hartree,
                          metadata=dict(parent.metadata,model_level='ZAFF4',
                            reference_coordinates_angstrom=np.asarray(reference).tolist(),
                            local_source_reconstruction=report))
    return field,report


def native_bonded_internal_derivatives(field):
    """Expand the actual bonded parent in its primitive coordinates to order 4.

    Products, reference shifts and every defining-bond switch are included.
    Only the resident one-dimensional library functions are differentiated
    numerically for F3/F4 (two step sizes and Richardson extrapolation).
    No off-reference electronic observation or Cartesian QFF is constructed.
    """
    from matrix_chem.primitive_force_field import (
        evaluate_primitive_functional_parameter, evaluate_torsion_fourier_parameter)
    q0 = np.asarray(field.field.equilibrium_values)
    polynomial = {}
    extrapolation = []

    def accumulate(key, value):
        polynomial[key] = polynomial.get(key, 0.) + float(value)

    def multiply(left, right):
        product = {}
        for key1,value1 in left.items():
            for key2,value2 in right.items():
                if len(key1)+len(key2) <= 4:
                    key = tuple(sorted(key1+key2))
                    product[key] = product.get(key,0.) + value1*value2
        return product

    def factor_series(factor):
        i,n = factor.primitive_index,factor.harmonic
        kind = field.primitives[i].kind
        if factor.form == 'outer_window':
            reference_vanishing_order((factor,), field.primitives)
            return {}
        if factor.form == 'inverse_distance':
            coefficients = (1., -1/q0[i], 1/q0[i]**2, -1/q0[i]**3)
        elif factor.form == 'native' and kind == 'bond':
            parameter = field.functions[field.field.primitive_type_ids[i]]
            if parameter.form == 'inverse_power_stretch':
                p = dict(parameter.parameters)['P']
                coefficients = (1., -(p+1)/(2*q0[i]),
                                (p+1)*(p+2)/(6*q0[i]**2),
                                -(p+1)*(p+2)*(p+3)/(24*q0[i]**3))
            else:
                alpha = field.couplings.alpha[i]
                coefficients = (1., -alpha/2, alpha**2/6, -alpha**3/24)
        elif factor.form == 'native' and kind == 'angle':
            if abs(np.sin(q0[i])) < 1e-6:
                raise ValueError('linear angle requires a nonsingular internal chart')
            cotangent = 1/np.tan(q0[i])
            coefficients = (1., cotangent/2, -1/6, -cotangent/24)
        elif factor.form in ('native','sin'):
            coefficients = (float(n),0.,-n**3/6,0.)
        elif factor.form == 'one_minus_cos':
            coefficients = (0.,n**2/2,0.,-n**4/24)
        elif factor.form == 'jet_sin':
            coefficients = (0.,0.,(-n**3+n)/6,0.)
        elif factor.form == 'jet_cos':
            coefficients = (0.,0.,0.,(n**4-n**2)/24)
        else:
            raise ValueError('unsupported parent factor for internal jet extraction')
        base = {(i,)*degree:value for degree,value in enumerate(coefficients,1) if value}
        answer = {():1.}
        for _ in range(factor.power):
            answer = multiply(answer,base)
        return answer

    for i,primitive in enumerate(field.primitives):
        periodic = primitive.kind == 'dihedral'
        parameter = (field.torsions if periodic else field.functions)[field.field.primitive_type_ids[i]]
        evaluator = evaluate_torsion_fourier_parameter if periodic else evaluate_primitive_functional_parameter
        energy,gradient,hessian = evaluator(parameter,q0[i])
        step = 1e-3 * (q0[i] if primitive.kind == 'bond' else 1.)
        estimates = []
        for h in (step,step/2):
            plus = evaluator(parameter,q0[i]+h)[2]
            minus = evaluator(parameter,q0[i]-h)[2]
            estimates.append(np.array(((plus-minus)/(2*h),(plus+minus-2*hessian)/h**2)))
        third,fourth = (4*estimates[1]-estimates[0])/3
        extrapolation.append(float(np.max(abs(estimates[1]-estimates[0]))/3))
        for degree,value in enumerate((gradient,hessian,third,fourth),1):
            accumulate((i,)*degree,value/factorial(degree))
        if primitive.kind != 'bond':
            for bond in field.couplings.supports[i]:
                accumulate((bond,)*4,-energy*field.couplings.alpha[bond]**4)

    products = [AsymptoticPicCorrection(c.force_constant,
                tuple(AsymptoticPicFactor(i,'native') for i in c.primitive_indices))
                for c in field.field.retained_couplings] + list(field.corrections)
    for correction in products:
        series = {():correction.coefficient}
        for factor in correction.factors:
            series = multiply(series,factor_series(factor))
        for key,value in series.items():
            if not key:
                raise ValueError('reference-constant correction needs explicit switch expansion')
            accumulate(key,value)
    derivatives = []
    for degree in range(1,5):
        derivatives.append({key:float(value*np.prod([factorial(p) for p in Counter(key).values()]))
                            for key,value in polynomial.items() if len(key)==degree and value})
    return derivatives, dict(schema='matrix.architect.native_internal_parent_jet.v1',
                             scope='bonded; frozen nonbonded excluded',
                             maximum_1D_extrapolation_change=float(max(extrapolation,default=0.)),
                             cartesian_or_electronic_points_used=False)


def _table(values, degree, count):
    result = {}
    for raw, value in values.items():
        key = tuple(sorted(raw))
        if (len(key) != degree or any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                                     or not 0 <= i < count for i in key)
                or key in result or not np.isfinite(value)):
            raise ValueError('invalid or duplicate internal force constant')
        if degree == 4 and len(set(key)) == 4 and value != 0:
            raise ValueError('four-distinct internal quartics must be zero at input')
        result[key] = float(value)
    return result


def fit_native_internal_quartic(parent, reference, cubic, quartic, *,
                                baseline_cubic, baseline_quartic):
    """Match a zero-closed internal bonded F3/F4, in Eh/angstrom/radian units.

    Baseline maps are the bonded parent's INTERNAL derivatives in the identical
    primitive gauge. They are required explicitly: modal observations, and the
    nonbonded field, cannot silently be substituted for these maps. Unlisted
    target entries are zero; every nonzero baseline entry must be supplied so
    that it can be cancelled when absent from the target. Native nonlinear
    coordinates' contribution from F3 to F4 is included exactly.
    """
    charge_source = str(getattr(parent.nonbonded_evaluator,'charge_source',''))
    if 'APOC' not in charge_source or 'CM5' not in charge_source:
        raise ValueError('ZAFF4 requires the frozen KEYMAKER/APOC CM5 baseline')
    count = len(parent.primitives)
    targets = [_table(cubic, 3, count), _table(quartic, 4, count)]
    baseline = [_table(baseline_cubic, 3, count), _table(baseline_quartic, 4, count)]
    q0 = np.asarray(parent.field.equilibrium_values)
    q = eval_primitives(parent.primitives, np.asarray(reference, float))
    if not np.allclose(q, q0, atol=1e-9, rtol=0):
        raise ValueError('internal quartic fit requires the actual primitive reference')
    residual = [{k: target.get(k, 0.) - base.get(k, 0.) for k in target.keys() | base.keys()}
                for target, base in zip(targets, baseline)]
    curvature = np.zeros(count)
    for i, primitive in enumerate(parent.primitives):
        if primitive.kind == 'bond':
            if q0[i] <= 0:
                raise ValueError('positive reference bond required')
            curvature[i] = -1. / q0[i]
        elif primitive.kind == 'angle':
            if abs(np.sin(q0[i])) < 1e-6:
                raise ValueError('linear angles require a nonsingular internal chart')
            curvature[i] = .5 / np.tan(q0[i])
        elif primitive.kind not in ('dihedral', 'out_of_plane'):
            raise ValueError('unsupported internal primitive')

    def term(key):
        return tuple(AsymptoticPicFactor(i, 'inverse_distance' if parent.primitives[i].kind == 'bond'
                                        else 'native' if parent.primitives[i].kind == 'angle'
                                        else 'sin', power)
                     for i, power in sorted(Counter(key).items()))

    def multiplicity(key):
        return np.prod([factorial(power) for power in Counter(key).values()])

    corrections = []
    for key, value in sorted(residual[0].items()):
        coefficient = value / multiplicity(key)
        if coefficient == 0:
            continue
        corrections.append(AsymptoticPicCorrection(float(coefficient), term(key)))
        for i, power in Counter(key).items():
            fourth = tuple(sorted((*key, i)))
            generated = coefficient * power * curvature[i] * multiplicity(fourth)
            residual[1][fourth] = residual[1].get(fourth, 0.) - generated
    for key, value in sorted(residual[1].items()):
        if value:
            corrections.append(AsymptoticPicCorrection(float(value / multiplicity(key)), term(key)))
    report = dict(schema='matrix.architect.internal_quartic_fit.v1',
                  coordinate_representation='internal_primitives',
                  derivative_units='hartree / angstrom^bond_count / radian^angular_count',
                  missing_internal_derivatives='zero_model_closure_before_fit',
                  four_distinct_internal_quartics=0,
                  nonbonded_changed=False, external_points_used=False,
                  fitted_correction_terms=len(corrections),
                  normal_mode_transformation='pending_asymptotics_and_minima_audit',
                  baseline_derivatives_supplied_by_caller=True)
    metadata = dict(parent.metadata, model_level='ZAFF4', internal_quartic_fit=report,
                    reference_coordinates_angstrom=np.asarray(reference).tolist(),
                    construction_status='INTERNAL_QUARTIC_MATCHED_PENDING_COMPLETION')
    field = NativePicField(parent.primitives, parent.field, parent.nonbonded_evaluator,
                           corrections=parent.corrections + tuple(corrections),
                           energy_offset_hartree=parent.energy_offset_hartree, metadata=metadata)
    return field, report


def reference_vanishing_order(factors, primitives):
    """Algebraic lower bound on vanishing order, including periodic factors."""
    order = 0
    protected = False
    for factor in factors:
        if (isinstance(factor.power, bool) or factor.power < 1 or int(factor.power) != factor.power
                or not 0 <= factor.primitive_index < len(primitives)
                or factor.harmonic < 1 or int(factor.harmonic) != factor.harmonic):
            raise ValueError('invalid native completion factor')
        if factor.form == 'outer_window':
            if (primitives[factor.primitive_index].kind != 'dihedral'
                    or not 0 < factor.window_start < factor.window_end < np.pi):
                raise ValueError('invalid protected periodic window')
            protected = True
            continue
        orders = {'native': 1, 'inverse_distance': 1, 'sin': 1,
                  'one_minus_cos': 2, 'jet_sin': 3, 'jet_cos': 4}
        if factor.form not in orders:
            raise ValueError('unsupported native completion factor')
        if factor.form in ('jet_sin', 'jet_cos') and factor.harmonic == 1:
            raise ValueError('identically zero Fourier completion factor')
        order += factor.power * orders[factor.form]
    return inf if protected else order


def complete_native_asymptotics(field, corrections, *, condition_roles=()):
    """Add only zero-four-jet native terms, with resident bond switches.

    External conditions, when supplied, concern angular boundaries or dihedral
    periodicity only. Their coefficients cannot enter the frozen quartic fit.
    All additions are serialized in the ordinary native runtime format.
    """
    allowed = {'angular_boundary', 'dihedral_periodicity', 'spurious_minimum_regularization'}
    if set(condition_roles) - allowed:
        raise ValueError('external E/G/H cannot recalibrate the local quartic field')
    corrections = tuple(corrections)
    for correction in corrections:
        if (not correction.factors or not np.isfinite(correction.coefficient)
                or reference_vanishing_order(correction.factors, field.primitives) < 5):
            raise ValueError('completion would change a derivative of order <= 4')
    report = dict(schema='matrix.architect.native_asymptotic_completion.v1',
                  added_terms=len(corrections), reference_derivatives_preserved_through_order=4,
                  proof='product vanishing order >= 5; smooth defining-bond switches',
                  condition_roles=list(condition_roles), local_quartic_refitted=False)
    metadata = dict(field.metadata, asymptotic_completion=report)
    return NativePicField(field.primitives, field.field, field.nonbonded_evaluator,
                          corrections=field.corrections + corrections,
                          energy_offset_hartree=field.energy_offset_hartree, metadata=metadata), report


def audit_native_minima(field, reference, search_basis_angstrom, seeds, bounds, *,
                        allowed_minima, gradient_tolerance=1e-6,
                        curvature_tolerance=1e-7, basin_tolerance=1e-3,
                        maximum_iterations=200, stop_on_failure=False,
                        refinement_iterations=0):
    """Finite multistart search on the native PES only; no electronic requests.

    The caller supplies a full vibrational Cartesian search chart and declares
    known physical basins. An unlisted minimum is a *candidate* spurious basin.
    Boundary hits, nonstationary endpoints and saddles remain unresolved.
    Classification uses the measured gradient and full projected Hessian;
    the optimizer's status flag is recorded separately from that certificate.
    """
    reference = np.asarray(reference, float)
    basis = np.asarray(search_basis_angstrom, float)
    seeds = np.asarray(seeds, float)
    limits = np.asarray(bounds, float)
    known = np.asarray(allowed_minima, float)
    if (reference.ndim != 2 or reference.shape[1] != 3 or basis.ndim != 2
            or basis.shape[1] != reference.size or not len(basis)
            or seeds.ndim != 2 or seeds.shape[1] != len(basis) or not len(seeds)
            or limits.shape != (len(basis), 2) or np.any(limits[:, 0] >= limits[:, 1])
            or known.ndim != 2 or known.shape[1] != len(basis) or not len(known)
            or not all(np.all(np.isfinite(v)) for v in (reference, basis, seeds, limits, known))
            or np.linalg.matrix_rank(basis) != len(basis)
            or np.any(seeds < limits[:, 0]) or np.any(seeds > limits[:, 1])):
        raise ValueError('finite, independent search chart, starts, bounds and known basins required')
    geometry_rank = np.linalg.matrix_rank(reference - reference.mean(axis=0))
    vibrational_dimension = reference.size - (5 if geometry_rank == 1 else 6)
    if len(basis) != vibrational_dimension:
        raise ValueError('minima classification requires a full vibrational search chart')
    if min(gradient_tolerance, curvature_tolerance, basin_tolerance) <= 0 or maximum_iterations < 1 or refinement_iterations < 0:
        raise ValueError('positive minima search tolerances and budget required')
    rows = []
    evaluator = getattr(field,'energy_gradient_vectorized',field.energy_gradient)
    reference_energy, _ = evaluator(reference)
    for start in seeds:
        last = {}
        def objective(z):
            energy, gradient = evaluator(reference + (z @ basis).reshape(reference.shape))
            value=energy-reference_energy
            derivative=basis @ np.asarray(gradient).reshape(-1) / BOHR_TO_ANGSTROM
            if not np.isfinite(value) or not np.all(np.isfinite(derivative)):
                raise FloatingPointError('nonfinite native E/G during minima search')
            last.update(coordinates=np.asarray(z).copy(),energy=float(value),gradient=derivative.copy())
            return value,derivative
        row=dict(search_coordinates=start.tolist(),relative_energy_hartree=None,
                 projected_gradient_max=None,minimum_projected_curvature=None,
                 optimizer_success=False,boundary=False,candidate_spurious_minimum=False,
                 candidate_unphysical_descent=False,resolved_known_minimum=False)
        try:
            result = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=limits,
                              options=dict(gtol=gradient_tolerance * .1, ftol=1e-14,
                                           maxiter=maximum_iterations))
            if (not result.success and refinement_iterations
                    and np.max(abs(result.jac)) > gradient_tolerance):
                # Restart the optimizer without relaxing any physical or
                # numerical threshold. A second failure remains unresolved.
                result = minimize(objective,result.x,jac=True,method='L-BFGS-B',bounds=limits,
                                  options=dict(gtol=gradient_tolerance*.1,ftol=1e-14,
                                               maxiter=refinement_iterations))
            energy, gradient = objective(result.x)
            boundary = bool(np.any(np.minimum(result.x-limits[:, 0], limits[:, 1]-result.x) < 1e-6))
            stationary = bool(np.max(abs(gradient)) <= gradient_tolerance)
            recognized = bool(min(np.linalg.norm(result.x - point) for point in known) <= basin_tolerance)
            row.update(search_coordinates=result.x.tolist(),relative_energy_hartree=float(energy),
                       projected_gradient_max=float(np.max(abs(gradient))),
                       optimizer_success=bool(result.success),boundary=boundary,
                       candidate_unphysical_descent=bool(energy < -1e-8 and not recognized))
            # A nonstationary boundary hit is already a failed search. Computing
            # its Hessian can encounter undefined angles and cannot certify it.
            if stationary and not boundary:
                hessian = field(reference + (result.x @ basis).reshape(reference.shape)).hessian_hartree_per_bohr2
                curvature=float(np.linalg.eigvalsh(basis @ hessian @ basis.T / BOHR_TO_ANGSTROM**2)[0])
                minimum=curvature > curvature_tolerance
                row.update(minimum_projected_curvature=curvature,
                           candidate_spurious_minimum=bool(minimum and not recognized),
                           resolved_known_minimum=bool(minimum and recognized),
                           independently_verified_stationary_point=True)
        except (FloatingPointError,np.linalg.LinAlgError) as error:
            row['numerical_failure']=str(error)
            if last:
                row.update(search_coordinates=last['coordinates'].tolist(),
                           relative_energy_hartree=last['energy'],
                           projected_gradient_max=float(np.max(abs(last['gradient']))),
                           candidate_unphysical_descent=bool(last['energy'] < -1e-8))
        rows.append(row)
        print('native minima audit', len(rows), 'of', len(seeds), rows[-1], flush=True)
        if stop_on_failure and not row['resolved_known_minimum']:
            break
    return dict(schema='matrix.architect.native_minima_audit.v1', points=rows,
                starts_requested=len(seeds),starts_completed=len(rows),
                search_passed=len(rows)==len(seeds) and all(row['resolved_known_minimum'] for row in rows),
                candidate_spurious_minima=sum(row['candidate_spurious_minimum'] for row in rows),
                candidate_unphysical_descents=sum(row['candidate_unphysical_descent'] for row in rows),
                external_points_used=False, global_absence_of_other_minima_proved=False,
                classification_basis='measured gradient, projected Hessian and declared physical basin; optimizer status recorded independently',
                scope='declared finite search chart, starts, bounds and physical basins')


def regularize_native_minima(field, reference, search_basis_angstrom, seeds, bounds, *,
                             allowed_minima, strengths_hartree,
                             preserved_condition_geometries=(), condition_tolerance=1e-10,
                             **search_options):
    """Select the first passing positive sixth-order completion on the same search.

    Every trial starts from the identical frozen quartic field. Strengths are
    an explicit caller policy; no source derivative is reweighted or refitted.
    A failed search returns no accepted field. It cannot trigger more QM work.
    """
    strengths = tuple(strengths_hartree)
    if (not strengths or any(not np.isfinite(v) or v < 0 for v in strengths)
            or list(strengths) != sorted(set(strengths)) or strengths[0] != 0):
        raise ValueError('explicit increasing strengths starting with zero required')
    terms = []
    for i, primitive in enumerate(field.primitives):
        if primitive.kind == 'bond':
            scale = 1. / field.couplings.alpha[i]
            terms.append((scale**-6, AsymptoticPicFactor(i, 'inverse_distance', 6)))
        elif primitive.kind == 'angle':
            terms.append((1., AsymptoticPicFactor(i, 'native', 6)))
        elif primitive.kind in ('dihedral', 'out_of_plane'):
            terms.append((1., AsymptoticPicFactor(i, 'one_minus_cos', 3)))
        else:
            raise ValueError('unsupported primitive for protected regularization')
    history = []
    conditions = tuple(np.asarray(xyz, float) for xyz in preserved_condition_geometries)
    if (not np.isfinite(condition_tolerance) or condition_tolerance <= 0
            or any(xyz.shape != np.shape(reference) or not np.all(np.isfinite(xyz)) for xyz in conditions)):
        raise ValueError('invalid protected angular/periodic conditions')
    condition_reference = [field.energy_gradient(xyz) for xyz in conditions]
    search_options.setdefault('stop_on_failure',True)
    for strength in strengths:
        print('native minima regularization strength_hartree',strength,flush=True)
        additions = tuple(AsymptoticPicCorrection(float(strength * scale), (factor,))
                          for scale, factor in terms) if strength else ()
        candidate, completion = complete_native_asymptotics(
            field, additions, condition_roles=('spurious_minimum_regularization',))
        condition_change = 0.
        for xyz, (energy0, gradient0) in zip(conditions, condition_reference):
            energy, gradient = candidate.energy_gradient(xyz)
            condition_change = max(condition_change, abs(energy-energy0),
                                   float(np.max(abs(gradient-gradient0))))
        if condition_change > condition_tolerance:
            history.append(dict(strength_hartree=float(strength),
                                rejected='ANGULAR_OR_PERIODIC_CONDITION_CHANGED',
                                condition_change_max=condition_change))
            continue
        audit = audit_native_minima(candidate, reference, search_basis_angstrom, seeds, bounds,
                                    allowed_minima=allowed_minima, **search_options)
        history.append(dict(strength_hartree=float(strength), audit=audit, completion=completion))
        if audit['search_passed']:
            report = dict(accepted=True, history=history, local_quartic_refitted=False,
                          normal_mode_transformation_allowed=True,
                          global_absence_of_other_minima_proved=False)
            candidate.metadata['minima_regularization'] = report
            return candidate, report
        if not audit['candidate_spurious_minima'] and not audit['candidate_unphysical_descents']:
            break  # Numerical failure without a physical counterexample is not a fitting target.
    return None, dict(accepted=False, history=history, local_quartic_refitted=False,
                      normal_mode_transformation_allowed=False,
                      global_absence_of_other_minima_proved=False)


def transform_internal_quartic(harmonic, cubic, quartic, first, second, third):
    """Stationary chain rule AFTER completion, with all induced modal entries.

    Coordinate jets differentiate internal displacements with respect to the
    requested normal coordinates. They include geometrical curvature through
    third order. These are potential derivatives, not factorial-divided Taylor
    coefficients. No four-distinct modal entry is discarded or fitted here.
    """
    f2, f3, f4, a, b, c = (np.asarray(v, float) for v in
                           (harmonic, cubic, quartic, first, second, third))
    p, n = a.shape if a.ndim == 2 else (0, 0)
    if (not p or not n or f2.shape != (p,)*2 or f3.shape != (p,)*3
            or f4.shape != (p,)*4 or b.shape != (p,n,n) or c.shape != (p,n,n,n)
            or not all(np.all(np.isfinite(v)) for v in (f2,f3,f4,a,b,c))):
        raise ValueError('matching internal tensors and coordinate jets required')
    modal2 = a.T @ f2 @ a
    t = np.einsum('ab,ai,bjk->ijk', f2, a, b, optimize=True)
    modal3 = np.einsum('abc,ai,bj,ck->ijk', f3, a, a, a, optimize=True)
    modal3 += t + t.transpose(1,0,2) + t.transpose(1,2,0)
    modal4 = np.einsum('abcd,ai,bj,ck,dl->ijkl', f4, a, a, a, a, optimize=True)
    t = np.einsum('abc,ai,bj,ckl->ijkl', f3, a, a, b, optimize=True)
    modal4 += (t + t.transpose(0,2,1,3) + t.transpose(0,3,2,1)
               + t.transpose(2,1,0,3) + t.transpose(3,1,2,0) + t.transpose(2,3,0,1))
    t = np.einsum('ab,aij,bkl->ijkl', f2, b, b, optimize=True)
    modal4 += t + t.transpose(0,2,1,3) + t.transpose(0,3,2,1)
    t = np.einsum('ab,ai,bjkl->ijkl', f2, a, c, optimize=True)
    modal4 += t + t.transpose(1,0,2,3) + t.transpose(1,2,0,3) + t.transpose(1,2,3,0)
    return modal2, modal3, modal4


def run_internal_quartic_protocol(request_path):
    """Portable KEYMAKER owner: internal fit -> completion -> minima -> modes."""
    from pathlib import Path
    from itertools import permutations
    import hashlib, json, resource, socket
    from threadpoolctl import threadpool_limits
    from matrix_core import resolve_keymaker_protocol
    from .native_pic_field import load_native_pic_field, save_native_pic_field
    path = Path(request_path).resolve()
    request = json.loads(path.read_text())
    required = {'schema','required_host','memory_gib','parent_field','internal_derivatives',
                'minima_search','strengths_hartree','output_directory'}
    optional = {'workers','asymptotic_corrections','normal_coordinate_jets','search_options'}
    if (request.get('schema') != 'matrix.architect.zaff4_internal_request.v1'
            or not required <= request.keys() or request.keys()-required-optional):
        raise ValueError('unsupported or incomplete internal ZAFF4 request')
    protocol = resolve_keymaker_protocol('zaff4-internal-asymptotic.v1')
    if (request['required_host'].split('.')[0] != socket.gethostname().split('.')[0]
            or request['memory_gib'] != 128):
        raise ValueError('declared remote host and 128 GiB required')
    resource.setrlimit(resource.RLIMIT_AS, (128*1024**3,)*2)
    workers = request.get('workers',4)
    if isinstance(workers,bool) or not isinstance(workers,int) or not 1 <= workers <= 128:
        raise ValueError('invalid worker budget')
    root = path.parent
    source_hashes = {}
    def source(name):
        value = (root/request[name]).resolve()
        source_hashes[name] = hashlib.sha256(value.read_bytes()).hexdigest()
        return value
    parent_path = source('parent_field')
    parent = load_native_pic_field(parent_path)
    with np.load(source('internal_derivatives'), allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata_json']))
        if (meta.get('schema') != 'matrix.architect.internal_bonded_derivatives.v1'
                or meta.get('parent_sha256') != source_hashes['parent_field']
                or meta.get('derivative_scope') != 'bonded_internal'
                or meta.get('missing_internal_derivatives') != 'zero_model_closure'
                or not meta.get('source_sha256')):
            raise ValueError('parent-bound INTERNAL bonded derivative provenance required')
        reference = archive['reference_angstrom']
        harmonic = archive['harmonic_internal'] if 'harmonic_internal' in archive else None
        maps = []
        for prefix in ('target',):
            pair = []
            for degree in (3,4):
                indices = archive[f'{prefix}_indices_{degree}']
                values = archive[f'{prefix}_values_{degree}_hartree']
                if (indices.shape != (len(values),degree) or indices.dtype.kind not in 'iu'
                        or len({tuple(sorted(row)) for row in indices}) != len(indices)):
                    raise ValueError('invalid or duplicate internal derivative rows')
                pair.append({tuple(row):float(v) for row,v in zip(indices,values)})
            maps.append(pair)
    with np.load(source('minima_search'), allow_pickle=False) as archive:
        search = {key:archive[key] for key in ('search_basis_angstrom','seeds','bounds','allowed_minima')}
        protected = archive['preserved_condition_geometries'] if 'preserved_condition_geometries' in archive else ()
    corrections = []
    roles = ()
    if request.get('asymptotic_corrections'):
        payload = json.loads(source('asymptotic_corrections').read_text())
        if payload.get('schema') != 'matrix.architect.native_asymptotic_conditions.v1':
            raise ValueError('typed angular/periodic completion required')
        roles = tuple(payload['condition_roles'])
        corrections = [AsymptoticPicCorrection(row['coefficient_hartree'],
                       tuple(AsymptoticPicFactor(**factor) for factor in row['factors']))
                       for row in payload['corrections']]
    output = (root/request['output_directory']).resolve()
    if output.exists():
        raise FileExistsError('internal ZAFF4 output must be fresh')
    output.mkdir(parents=True)
    with threadpool_limits(limits=workers):
        parent_jets, baseline_report = native_bonded_internal_derivatives(parent)
        baseline = parent_jets[2:]
        n = len(parent.primitives)
        parent_harmonic = np.zeros((n,n))
        for key,value in parent_jets[1].items():
            for permuted in set(permutations(key)):
                parent_harmonic[permuted] = value
        if harmonic is not None and not np.allclose(harmonic,parent_harmonic,atol=1e-10,rtol=1e-10):
            raise ValueError('supplied harmonic field differs from the frozen bonded parent')
        harmonic = parent_harmonic
        fitted, local = fit_native_internal_quartic(parent, reference, *maps[0],
                           baseline_cubic=baseline[0], baseline_quartic=baseline[1])
        fitted_jets, _ = native_bonded_internal_derivatives(fitted)
        errors = []
        for expected,actual in zip(parent_jets[:2] + maps[0], fitted_jets):
            errors.append(max((abs(expected.get(key,0.)-actual.get(key,0.))
                               for key in expected.keys() | actual.keys()), default=0.))
        local['maximum_internal_derivative_error_by_order'] = errors
        local['local_derivatives_passed'] = all(error <= 1e-9 for error in errors)
        if not local['local_derivatives_passed']:
            report = dict(schema='matrix.architect.internal_quartic_workflow.v1',
                          accepted=False, internal_fit=local, baseline=baseline_report,
                          failure='INTERNAL_DERIVATIVE_MATCH_FAILED',
                          external_QM_points_acquired=0)
            (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            return report
        save_native_pic_field(output/'zaff4-internal-fitted.json', fitted)
        completed, completion = complete_native_asymptotics(fitted, corrections, condition_roles=roles)
        save_native_pic_field(output/'zaff4-asymptotic-candidate.json', completed)
        selected, regularization = regularize_native_minima(completed, reference,
                       **search, strengths_hartree=request['strengths_hartree'],
                       preserved_condition_geometries=protected, **request.get('search_options',{}))
        modal_exported = False
        if selected is not None:
            save_native_pic_field(output/'zaff4-minima-checked.json', selected)
            if request.get('normal_coordinate_jets'):
                tensors = []
                for degree, table in zip((3,4), maps[0]):
                    tensor = np.zeros((n,)*degree)
                    for key,value in table.items():
                        for permuted in set(permutations(key)):
                            tensor[permuted] = value
                    tensors.append(tensor)
                with np.load(source('normal_coordinate_jets'), allow_pickle=False) as jets:
                    if (str(jets['parent_sha256']) != source_hashes['parent_field']
                            or not np.array_equal(jets['reference_angstrom'], reference)):
                        raise ValueError('normal coordinate jets belong to another reference')
                    # The bonded contribution need not be stationary by itself:
                    # frozen nonbonded forces cancel its reference gradient.
                    # A linear internal-normal map needs no omitted gradient
                    # times fourth-coordinate-derivative contribution.
                    if ('coordinate_map' not in jets
                            or str(jets['coordinate_map']) != 'linear_internal_modes'
                            or np.any(jets['second']) or np.any(jets['third'])):
                        raise ValueError('bonded modal export requires a declared linear internal-normal map')
                    arrays = transform_internal_quartic(harmonic, *tensors,
                                                        jets['first'],jets['second'],jets['third'])
                np.savez_compressed(output/'normal-bonded-quartic.npz', harmonic=arrays[0],
                                    cubic=arrays[1], quartic=arrays[2],
                                    scope='bonded contribution; frozen nonbonded remains separate',
                                    four_distinct_modal_terms_retained=True)
                modal_exported = True
    report = dict(schema='matrix.architect.internal_quartic_workflow.v1',
                  protocol_id=protocol.protocol_id, protocol_sha256=protocol.sha256,
                  execution_host=socket.gethostname(), memory_gib=128,
                  request_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  source_sha256=source_hashes, internal_fit=local, baseline=baseline_report, asymptotic_completion=completion,
                  minima_regularization=regularization, accepted=selected is not None,
                  normal_bonded_quartic_exported=modal_exported, external_QM_points_acquired=0,
                  workflow_order=['internal_zero_closure','internal_ZAFF_fit',
                                  'asymptotic_completion','minima_audit_and_regularization',
                                  'normal_mode_transformation_if_accepted'])
    (output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report
