"""Resolve Gaussian printed-force-field phases against a harmonic FCHK gauge.

Anharmonic inertia derivatives provide signed phase anchors. Archived Hessian
stencils determine relative phases for modes whose diagonal inertia derivative
vanishes by symmetry. No electronic calculation or observable fitting is used.
"""
from hashlib import sha256
from pathlib import Path
import re

import numpy as np

from matrix_rovib.isotopic_cubic import rovibrational_kinematics_from_modes
from matrix_rovib.vibin import modes_from_hessian
from .fchk import read_gaussian_fchk, lower_to_symmetric, BOHR_TO_ANGSTROM
from .parsers import _parse_cartesian_hessian_block, _parse_last_orientation


def align_gaussian_normal_field(log_path, fchk_path, normal_derivatives, *, anharmonic_to_harmonic):
    """Return full tensors in the exact FCHK harmonic gauge, plus provenance.

    Input normal tensors use physical mass-weighted coordinates (not reduced
    frequency-scaled coordinates). All supplied orders receive the same phase
    change. Unanchored sign groups are permitted only when every supplied
    tensor is invariant under the remaining simultaneous sign flip.
    """
    log_path, fchk_path = Path(log_path), Path(fchk_path)
    lines = log_path.read_text().splitlines()
    fchk = read_gaussian_fchk(fchk_path)
    xyz = fchk.cartesian_coordinates_bohr.reshape((-1,3))*BOHR_TO_ANGSTROM
    masses = fchk.masses_amu
    w, modes = modes_from_hessian(masses, fchk.cartesian_hessian_lower, xyz)
    n = len(w); dimension = 3*len(masses)
    mapping = tuple(anharmonic_to_harmonic)
    if sorted(mapping) != list(range(1,n+1)):
        raise ValueError('complete explicit Gaussian H/A mapping required')
    tensors = {rank:np.asarray(value,dtype=float) for rank,value in normal_derivatives.items()}
    if 3 not in tensors or any(v.shape != (n,)*k or not np.all(np.isfinite(v)) for k,v in tensors.items()):
        raise ValueError('complete finite normal tensors including F3 required')
    anchors = {}
    starts = [i for i,l in enumerate(lines) if 'Inertia Moments Derivatives w.r.t. Normal Modes' in l]
    if not starts:
        raise ValueError('Gaussian anharmonic inertia derivatives required for source gauge')
    source = np.zeros((n,3)); seen=set()
    for line in lines[starts[-1]+1:]:
        match=re.match(r'\s*Q\(\s*(\d+)\)\s+(.+)',line)
        if not match: continue
        a=int(match[1]); h=mapping[a-1]-1
        if h in seen:raise ValueError('duplicate inertia derivative mode')
        source[h]=np.asarray([float(v) for v in match[2].split()[:6]])[[0,2,5]]
        seen.add(h)
        if len(seen)==n:break
    if len(seen)!=n:raise ValueError('incomplete inertia derivative mode table')
    kin=rovibrational_kinematics_from_modes(xyz,masses,modes)
    target=kin.inertia_derivatives_amu_sqrt_ang[:3].T
    for i in range(n):
        if max(abs(source[i]))>1e-4:
            phase=int(np.sign(source[i]@target[i]))
            if phase==0 or not np.allclose(source[i]*phase,target[i],rtol=2e-4,atol=2e-5):
                raise ValueError('inertia derivatives and FCHK modes disagree beyond print precision')
            anchors[i]=phase
    edges=[]
    if len(anchors)<n:
        hstarts=[i for i,l in enumerate(lines) if 'Force constants in Cartesian coordinates' in l]
        markers=[]
        for i,line in enumerate(lines):
            match=re.search(r'Re-enter D2Numr:\s+normal mode\s+(\d+) step-(up|down)\s+number\s+1',line)
            if match:markers.append((i,int(match[1])-1,match[2]))
        if not markers or not hstarts:
            raise ValueError('unanchored mode phases require archived Hessian stencils')
        # D2Numr re-entry reports the point just completed, not the following one.
        stencils={}; previous=-1
        for line,mode,direction in markers:
            points=[s for s in hstarts if previous<s<line]
            if not points:raise ValueError('Hessian missing before D2Numr re-entry')
            key=(mode,direction)
            if key in stencils:raise ValueError('multiple Hessian stencils require an explicit selection')
            stencils[key]=_parse_cartesian_hessian_block(lines[points[-1]+1:],size=dimension)
            previous=line
        final=[s for s in hstarts if s>markers[-1][0]]
        if not final:raise ValueError('final equilibrium Hessian required to verify the Cartesian frame')
        eq=_parse_cartesian_hessian_block(lines[final[-1]+1:],size=dimension)
        reference=lower_to_symmetric(fchk.cartesian_hessian_lower)
        # Gaussian can print equilibrium Hessians in the input frame and the
        # displaced Hessians in the standard frame. Verify each independently.
        rotations=[np.eye(dimension)]
        orientation_lines=[i for i,l in enumerate(lines[:markers[0][0]])
                           if 'Input orientation:' in l or 'Standard orientation:' in l]
        centered=xyz-np.average(xyz,axis=0,weights=masses)
        for begin in orientation_lines:
            geometry=_parse_last_orientation(lines[:begin+len(masses)+8],source_path=log_path)
            if geometry is None or geometry.coordinates_angstrom.shape!=xyz.shape:continue
            original=geometry.coordinates_angstrom
            original=original-np.average(original,axis=0,weights=masses)
            u,_,vt=np.linalg.svd((original*masses[:,None]).T@centered)
            correction=np.eye(3);correction[-1,-1]=np.linalg.det(u@vt)
            rotation=u@correction@vt
            if np.max(abs(original@rotation-centered))<3e-5:
                rotations.append(np.kron(np.eye(len(masses)),rotation))
        eq_error=min(np.max(abs(R.T@eq@R-reference)) for R in rotations)
        if eq_error>5e-6:
            raise ValueError('archived equilibrium Hessian and FCHK differ beyond print precision')
        centers=[(stencils[i,'up']+stencils[i,'down'])/2 for i in range(n)
                 if (i,'up') in stencils and (i,'down') in stencils]
        if not centers:raise ValueError('paired Hessian stencils required')
        scores=[sum(np.linalg.norm(R.T@center@R-reference)**2 for center in centers)
                for R in rotations]
        R=rotations[int(np.argmin(scores))]
        if max(np.max(abs(R.T@center@R-reference)) for center in centers)>2e-3:
            raise ValueError('Hessian stencils do not share the verified parent Cartesian frame')
        stencils={key:R.T@value@R for key,value in stencils.items()}
        D=modes.reshape(n,dimension)/np.sqrt(np.repeat(masses,3))[None,:]
        f3=tensors[3]
        for i in range(n):
            if (i,'up') not in stencils or (i,'down') not in stencils:continue
            diff=D@(stencils[i,'up']-stencils[i,'down'])@D.T
            diagonal=np.diag(f3[i]); mask=abs(diagonal)>max(abs(diagonal))*1e-2
            if not np.any(mask) or np.linalg.norm(diagonal[mask])<1e-12:continue
            factor=np.dot(np.diag(diff)[mask],diagonal[mask])/np.dot(diagonal[mask],diagonal[mask])
            predicted=factor*f3[i]
            threshold=max(np.max(abs(diff)),np.max(abs(predicted)))*1e-3
            for j in range(n):
                for k in range(j+1,n):
                    if abs(predicted[j,k])<=threshold:continue
                    ratio=diff[j,k]/predicted[j,k]
                    if abs(abs(ratio)-1)<.05:
                        edges.append((j,k,int(np.sign(ratio))))
    graph={i:[] for i in range(n)}
    for j,k,sign in edges:
        graph[j].append((k,sign));graph[k].append((j,sign))
    phases=np.zeros(n,dtype=int);unanchored=[]
    while np.any(phases==0):
        available=np.where(phases==0)[0]
        seed=next((i for i in available if i in anchors),int(available[0]))
        phases[seed]=anchors.get(seed,1); stack=[seed];component=[]
        while stack:
            i=stack.pop();component.append(i)
            if i in anchors and phases[i]!=anchors[i]:
                raise ValueError('inconsistent source mode phases from inertia/Hessian evidence')
            for j,sign in graph[i]:
                value=phases[i]*sign
                if phases[j] and phases[j]!=value:
                    raise ValueError('inconsistent Hessian stencil phase constraints')
                if not phases[j]:phases[j]=value;stack.append(j)
        if not any(i in anchors for i in component):unanchored.append(component)
    # A symmetry-related simultaneous sign flip must leave the entire model invariant.
    for component in unanchored:
        flip=np.ones(n);flip[component]=-1
        for rank,tensor in tensors.items():
            changed=tensor.copy()
            for axis in range(rank):
                shape=[1]*rank;shape[axis]=n;changed*=flip.reshape(shape)
            if not np.allclose(changed,tensor,rtol=0,atol=1e-12):
                raise ValueError('source phase ambiguity changes the field; additional archived evidence required')
    aligned={}
    for rank,tensor in tensors.items():
        tensor=tensor.copy()
        for axis in range(rank):
            shape=[1]*rank;shape[axis]=n;tensor*=phases.reshape(shape)
        aligned[rank]=tensor
    return aligned,xyz,masses,w,modes,{
        'log_sha256':sha256(log_path.read_bytes()).hexdigest(),
        'fchk_sha256':sha256(fchk_path.read_bytes()).hexdigest(),
        'phases_harmonic_order':phases.tolist(), 'inertia_phase_anchors':len(anchors),
        'hessian_phase_constraints':len(edges),
        'field_invariant_unanchored_components':unanchored,
    }
