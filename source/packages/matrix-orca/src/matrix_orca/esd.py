"""Read phase-preserving harmonic/HT data written by ORCA ESD Hessian files.

ESD's mass-weighted eigenvectors define the derivative phase and ordering;
Re-diagonalization would lose that information. This importer never replaces
negative curvatures, rotates an already common frame, or calls electronics.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from .parsers import hessian_input_from_orca_hessian_file


def _section(lines,name):
    try:start=lines.index(name)+1
    except ValueError as error:raise ValueError('Missing ORCA ESD section '+name) from error
    end=next((i for i in range(start,len(lines)) if lines[i].startswith('$')),len(lines))
    return [line for line in lines[start:end] if line.strip() and not line.lstrip().startswith('#')]


def _blocked_matrix(lines,name):
    block=_section(lines,name);dimensions=block[0].split()
    if len(dimensions)!=2:raise ValueError('Two ORCA matrix dimensions required')
    n,m=map(int,dimensions)
    if min(n,m)<1 or max(n,m)>100000:raise ValueError('Invalid ORCA ESD matrix dimensions')
    matrix=np.zeros((n,m));seen=np.zeros((n,m),bool);columns=None
    for line in block[1:]:
        tokens=line.split()
        if all(re.fullmatch(r'\d+',x) for x in tokens):
            columns=list(map(int,tokens))
            if len(set(columns))!=len(columns) or any(j>=m for j in columns):raise ValueError('Repeated or out-of-bounds ORCA matrix column')
            continue
        if columns is None or len(tokens)!=len(columns)+1:raise ValueError('Malformed ORCA matrix block')
        row=int(tokens[0]);values=[float(x.replace('D','E')) for x in tokens[1:]]
        if not 0<=row<n or np.any(seen[row,columns]):raise ValueError('Repeated or invalid ORCA matrix row')
        matrix[row,columns]=values;seen[row,columns]=True
    if not np.all(seen) or not np.all(np.isfinite(matrix)):raise ValueError('Incomplete ORCA ESD matrix')
    return matrix


@dataclass(frozen=True)
class ESDHarmonicData:
    initial_frequencies_cm: np.ndarray
    final_frequencies_cm: np.ndarray
    mode_map: np.ndarray
    final_shift_dimensionless: np.ndarray
    final_gradient_dimensionless_au: np.ndarray
    reference_dipole_au: np.ndarray
    electronic_gap_cm: float
    input_time_step_fs: float
    gaussian_sigma_cm: float
    time_points: int


def read_esd_harmonic_data(ground_hessian,excited_hessian,output,*,electronic_gap_cm=None):
    """Import completed Cartesian AH/HT fields in ESD's stored common frame.

    Stored transition derivatives are per mass-weighted FINAL normal coordinate
    in atomic units, with alternating real/imaginary Cartesian components.
    They are converted to dimensionless modes by division by sqrt(omega_au).
    ESD's dimensionless displacement step is distinct from this storage unit.
    The reference dipole
    and adiabatic gap are taken from the final ESD correlation summary. An
    explicitly supplied original gap avoids that summary's two-decimal rounding;
    it must agree with the printed value to its displayed precision.
    """
    paths=[Path(ground_hessian),Path(excited_hessian)]
    fields=[hessian_input_from_orca_hessian_file(path) for path in paths]
    a,b=fields
    if not np.array_equal(a.atomic_numbers,b.atomic_numbers) or not np.array_equal(a.masses_amu,b.masses_amu):raise ValueError('Matching atom order and isotope masses required')
    lines=[path.read_text().splitlines() for path in paths]
    vectors=[_blocked_matrix(text,'$eigenvectors_mass_weighted_hessian') for text in lines]
    wi,wf=[np.asarray(field.harmonic_frequencies_cm) for field in fields];n=len(wi)
    if len(wf)!=n or np.any(wi<=0) or np.any(wf<=0):raise ValueError('Two stable equally sized mode spaces required')
    for modes in vectors:
        if modes.shape!=(3*len(a.masses_amu),n) or not np.allclose(modes.T@modes,np.eye(n),atol=2e-9,rtol=0):raise ValueError('Stored phase-resolved orthonormal modes required')
    derivative=_section(lines[1],'$transition_dipole_derivatives')
    if int(derivative[0])!=n or len(derivative)!=n+1:raise ValueError('One transition derivative per final mode required')
    values=np.array([[float(t.replace('D','E')) for t in row.split()] for row in derivative[1:]])
    if values.shape!=(n,6) or not np.all(np.isfinite(values)):raise ValueError('Six finite real/imaginary dipole columns required')
    h=(values[:,::2]+1j*values[:,1::2]).T*np.sqrt(219474.6313705/wf)[None,:]
    text=Path(output).read_text()
    if 'ORCA ESD FINISHED WITHOUT ERROR' not in text:raise ValueError('Completed ESD output required')
    summary=text[text.rindex('***Everything is set'):]
    number=r'[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?'
    block=summary[summary.index('Reference transition dipole'):summary.index('Calculating correlation function:')]
    pairs=re.findall(r'\(('+number+r')\s+('+number+r')\)',block)
    if len(pairs)!=3:raise ValueError('Three complex reference dipole components required')
    mu=np.array([float(x)+1j*float(y) for x,y in pairs])
    gap=float(re.search(r'Adiabatic energy difference:\s*('+number+')',summary)[1])
    if electronic_gap_cm is not None:
        if not np.isfinite(electronic_gap_cm) or abs(electronic_gap_cm-gap)>.0051:
            raise ValueError('Original gap must agree with the printed ESD summary')
        gap=float(electronic_gap_cm)
    window=float(re.search(r'Maximum time:\s*('+number+')',summary)[1]);points=int(re.search(r'Number of points:\s*(\d+)',summary)[1])
    sigma=float(re.search(r'Inhomogeneous linewidth is:\s*('+number+')',summary)[1])
    if not np.all(np.isfinite([gap,window,sigma])) or window<=0 or points<=0 or sigma<0:
        raise ValueError('Finite gap, positive time grid and nonnegative linewidth required')
    scale=5140.487143715055/np.sqrt(219474.6313705)
    delta=(np.sqrt(a.masses_amu)[:,None]*(a.cartesian_coordinates_bohr-b.cartesian_coordinates_bohr)).ravel()
    shift=np.sqrt(wf)*(vectors[1].T@delta)/scale
    return ESDHarmonicData(wi,wf,vectors[1].T@vectors[0],shift,h,mu,gap,window/points,sigma,points+1)


def audit_esd_derivative_displacements(data, output, *, dimensionless_step):
    """Check imported HT units against independently printed central dipoles.

    The caller supplies the actual displacement step. Printed dipole precision
    determines the achievable residual; this routine does not fit any scale.
    """
    if not np.isfinite(dimensionless_step) or dimensionless_step<=0:
        raise ValueError('Positive finite dimensionless displacement required')
    number=r'[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?'
    component=r'\(('+number+r')\s+('+number+r')\)'
    pattern=(r'<<Displacing mode (\d+) \(([+-])\)>>\s+tdip =\s*'
             +component+r',\s*'+component+r',\s*'+component)
    values={}
    for match in re.finditer(pattern,Path(output).read_text()):
        row=match.groups();key=(int(row[0])-1,row[1])
        if key in values:raise ValueError('Ambiguous repeated ESD displacement record')
        values[key]=np.array([float(row[j])+1j*float(row[j+1]) for j in (2,4,6)])
    n=len(data.final_frequencies_cm)
    if set(values)!={(i,sign) for i in range(n) for sign in ('+','-')}:
        raise ValueError('One complete central-displacement pair per mode required')
    numerical=np.array([(values[i,'+']-values[i,'-'])/(2*dimensionless_step) for i in range(n)]).T
    difference=float(np.max(abs(numerical-data.final_gradient_dimensionless_au)))
    maximum=float(np.max(abs(data.final_gradient_dimensionless_au)))
    return dict(central_displacements_read=len(values),dimensionless_step=float(dimensionless_step),
        maximum_derivative_difference_au=difference,
        maximum_relative_to_largest_derivative=None if maximum==0 else difference/maximum,
        fitted_scale=False,scope='Independent logged central-dipole check, limited by printed precision')
