"""Finite-source coupled torsional fitting through the registered KeyMaker route.

This owner delegates the numerical fit to periodic_hermite. It acquires no
points and does not claim a converged nuclear-motion spectrum from an energy
fit alone. The subsequent joint DVR belongs to LOCK's existing LAM solver.
"""
from pathlib import Path
import json

WORKFLOW_ID = 'coupled-local-torsions.v1'


def run_coupled_torsions_request(request_path):
    """Validate the two-phase contract and execute the existing fit owner."""
    from .periodic_hermite import run_periodic_hermite_request
    path = Path(request_path).resolve()
    request = json.loads(path.read_bytes())
    if (request.get('schema') != 'matrix.dvr.analytic_surface_request.v1'
            or request.get('family') != 'periodic_hermite'
            or request.get('workflow_id') != WORKFLOW_ID):
        raise ValueError('Typed coupled-local-torsions.v1 periodic request required')
    labels = request.get('phase_labels', [])
    if (not isinstance(labels, list) or len(labels) != 2
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != 2):
        raise ValueError('Two distinct physical torsional phase labels required')
    coordinates = request.get('coordinates_radian', [])
    if not coordinates or any(len(row) != 2 for row in coordinates):
        raise ValueError('Two physical angular coordinates per source configuration required')
    # Avoid selecting a functional form using the independent validation point.
    if len(request.get('candidates', [])) != 1:
        raise ValueError('Freeze one functional form before held-out validation')
    if not isinstance(request.get('global_reflection_even'), bool):
        raise ValueError('Declare reflection symmetry explicitly for the reactive channel')
    result_path = run_periodic_hermite_request(path)
    result = json.loads(result_path.read_bytes())
    result.update(workflow_id=WORKFLOW_ID, phase_labels=labels,
                  nuclear_motion_status='not_computed',
                  next_owner='matrix_rovib.lam_rotation.solve_lam_basis',
                  required_kinetic_input='full positive two-phase mass metric, including off-diagonal terms',
                  qualification_scope='Held-out potential energies only; joint DVR spectrum and rate convergence remain separate gates')
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    return result_path
