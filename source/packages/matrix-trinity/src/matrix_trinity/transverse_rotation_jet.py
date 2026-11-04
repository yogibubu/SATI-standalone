"""Reference-local transverse rotational jets on an unchanged nuclear space.

This explicit control retains the complete first rotational vertex derivative
and second rotational metric at the reference. The zero-J metric and measure
remain frozen in transverse displacement. It is not a full metric completion.
"""
from dataclasses import replace

import numpy as np
from scipy.sparse import csr_matrix


def complete_transverse_rotation_jet(coupled, active_momenta, active_coordinates,
        bath, kinetic_first_cm, rotational_second_cm):
    """Add b-linear rotational vertices and b, a*b, b*b rotational tensors.

    Inputs are true reference derivatives in active/bath/rotation order.
    Products retain the exterior oscillator intermediates owned by ``bath``.
    The caller must replace the harmonic counterterm with the same jets.
    """
    p=np.asarray(active_momenta);q=np.asarray(active_coordinates)
    first=np.asarray(kinetic_first_cm,float);second=np.asarray(rotational_second_cm,float)
    d=len(p);nb=len(bath['frequencies_cm']);nv=d+nb;M=coupled.active_dimension;B=coupled.bath_dimension
    if (p.shape!=(d,M,M) or q.shape!=p.shape or first.shape!=(nv+3,nv+3,nv)
            or second.shape!=(3,3,nv,nv) or tuple(bath['occupations'])!=tuple(coupled.occupations)
            or not all(np.all(np.isfinite(x)) for x in (p,q,first,second))
            or not np.allclose(p,p.conj().swapaxes(1,2)) or not np.allclose(q,q.conj().swapaxes(1,2))
            or not np.allclose(first,first.swapaxes(0,1))
            or not np.allclose(second,second.swapaxes(0,1)) or not np.allclose(second,second.swapaxes(2,3))):
        raise ValueError('Matching Hermitian active moments and complete reference rotational jets required')
    linear=[list(row) for row in coupled.linear_terms]
    quadratic=[[list(row) for row in axis] for axis in coupled.quadratic_terms]
    identity=np.eye(M);zero=lambda:csr_matrix((B,B),dtype=complex)
    def add(terms,a,b):
        if np.any(a) and b.nnz:terms.append((a,b))
    for axis in range(3):
        for s in range(d):
            coefficients=first[s,nv+axis,d:nv]@bath['coordinate_map']
            for b in range(nb):
                add(linear[axis],coefficients[b]*p[s],bath['normal_coordinate'][b])
        # Hermitian p*b, evaluated through the exact exterior intermediate.
        coefficients=np.einsum('bi,bk,kj->ij',bath['momentum_map'],first[d:nv,nv+axis,d:nv],bath['coordinate_map'])
        operator=zero()
        for i in range(nb):
            for j in range(nb):
                if coefficients[i,j]:
                    pq=bath['normal_momentum_coordinate'][i,j]
                    operator+=coefficients[i,j]*(pq+pq.conj().T)/2
        add(linear[axis],identity,operator)
        for other in range(3):
            terms=quadratic[axis][other];operator=zero()
            first_normal=first[nv+axis,nv+other,d:nv]@bath['coordinate_map']
            mixed_normal=second[axis,other,:d,d:nv]@bath['coordinate_map']
            second_normal=bath['coordinate_map'].T@second[axis,other,d:nv,d:nv]@bath['coordinate_map']
            for b in range(nb):
                operator+=.5*first_normal[b]*bath['normal_coordinate'][b]
                for s in range(d):
                    add(terms,.5*mixed_normal[s,b]*q[s],bath['normal_coordinate'][b])
                for k in range(nb):
                    if second_normal[b,k]:
                        operator+=.25*second_normal[b,k]*bath['normal_coordinate_second'][b,k]
            add(terms,identity,operator)
    return replace(coupled,linear_terms=tuple(tuple(row) for row in linear),
        quadratic_terms=tuple(tuple(tuple(row) for row in axis) for axis in quadratic))


def transverse_rotational_reference_jets(frozen_first_cm, full_first_cm, full_rotational_second_cm,
                                        active_dimensions):
    """Counterterm derivatives of precisely the same reference-local model."""
    frozen=np.asarray(frozen_first_cm,float);full=np.asarray(full_first_cm,float)
    n=full.shape[-1];d=active_dimensions
    if (frozen.shape!=(n+3,n+3,n) or full.shape!=frozen.shape or not 0<d<n
            or np.shape(full_rotational_second_cm)!=(3,3,n,n)):
        raise ValueError('Compatible full/frozen reference derivatives required')
    first=frozen.copy();first[n:,:,d:]=full[n:,:,d:];first[:,n:,d:]=full[:,n:,d:]
    return first,np.array(full_rotational_second_cm,copy=True)
