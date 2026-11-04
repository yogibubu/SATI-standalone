"""Coordinate-property tensor averages in the production n-mode VSCF states."""
from __future__ import annotations

import numpy as np


def _parse_expansion(payload, path_count, active_dimension, mode_count):
    active=np.asarray(payload['active_powers']);contracted=np.asarray(payload['contracted_powers'])
    if not np.isrealobj(payload['coefficients']):raise ValueError('VSCF tensor coefficients must be real')
    coefficients=np.asarray(payload['coefficients'],float)
    if (active.ndim!=2 or len(active)<1 or contracted.shape!=(len(active),mode_count) or active.shape[1]!=active_dimension
        or coefficients.ndim<2 or coefficients.shape[:2]!=(path_count,len(active))
        or not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(active))
        or not np.all(np.isfinite(contracted)) or np.any(active<0) or np.any(contracted<0)
        or np.any(active!=np.floor(active)) or np.any(contracted!=np.floor(contracted))
        or payload.get('coefficient_frame')!='reference' or payload.get('reference') not in {'absolute','increment'}):
        raise ValueError('invalid reference-frame VSCF property expansion')
    terms={}
    for k,(a,c) in enumerate(zip(active,contracted,strict=True)):
        key=tuple(map(int,a)),tuple(map(int,c))
        terms[key]=terms.get(key,0)+coefficients[:,k]
    return terms


def _multiply(left,right):
    result={}
    for (a,c),x in left.items():
        for (b,d),y in right.items():
            key=tuple(i+j for i,j in zip(a,b,strict=True)),tuple(i+j for i,j in zip(c,d,strict=True))
            result[key]=result.get(key,0)+x@y
    return result


def prepare_vscf_tensor_fields(surface,specification):
    """Normalize fields and form the complete kinetic polynomial before averaging."""
    raw=dict(specification.get('tensor_surfaces',{}));fields={};references={}
    mode_count=np.shape(specification['frequencies_cm'])[1]
    for name,payload in raw.items():
        if name not in {'F_MHz','rho','R_MHz','quadratic_kinetic_MHz','dipole_au','geometric_potential_MHz'} and not name.startswith('spin_tensor_surfaces_MHz.'):
            raise ValueError(f'unsupported VSCF tensor target: {name}')
        fields[name]=_parse_expansion(payload,surface.grid_shape[0],surface.dimension-1,mode_count)
        references[name]=payload['reference']
    kinetic={'F_MHz','rho','R_MHz'}
    if {'F_MHz','rho'} & fields.keys():
        if not kinetic <= fields.keys() or 'quadratic_kinetic_MHz' in fields or any(references[k]!='absolute' for k in kinetic):
            raise ValueError('fluctuating F/rho require complete absolute F, rho and R expansions')
        f,rho,r=(fields.pop(k) for k in ('F_MHz','rho','R_MHz'))
        d=surface.dimension
        if any(value.shape[1:]!=tail for terms,tail in ((f,(d,d)),(rho,(3,d)),(r,(3,3))) for value in terms.values()):
            raise ValueError('kinetic polynomial tensor dimensions do not match the LAM surface')
        rho_t={key:value.swapaxes(-1,-2) for key,value in rho.items()}
        mixed=_multiply(f,rho_t)
        rotation=dict(r)
        for key,value in _multiply(_multiply(rho,f),rho_t).items():rotation[key]=rotation.get(key,0)+value
        joint={}
        for key in set(f)|set(mixed)|set(rotation):
            matrix=np.zeros((surface.grid_shape[0],d+3,d+3))
            if key in f:matrix[:,:d,:d]=f[key]
            if key in mixed:matrix[:,:d,d:]=-mixed[key];matrix[:,d:,:d]=-mixed[key].swapaxes(-1,-2)
            if key in rotation:matrix[:,d:,d:]=rotation[key]
            joint[key]=matrix
        fields['quadratic_kinetic_MHz']=joint;references['quadratic_kinetic_MHz']='absolute'
    if 'quadratic_kinetic_MHz' in fields and references['quadratic_kinetic_MHz']!='absolute':
        raise ValueError('a joint kinetic expansion must describe the absolute quadratic form')
    if 'quadratic_kinetic_MHz' in fields and 'R_MHz' in fields:
        raise ValueError('do not supply R twice alongside a joint kinetic expansion')
    words=tuple(sorted({c for terms in fields.values() for _,c in terms}))
    return fields,references,words


def contract_vscf_tensor_fields(surface,fields,references,points,moment_record):
    """Return averaged fields; kinetic averaging retains cross correlations."""
    words={tuple(word):i for i,word in enumerate(moment_record['powers'])}
    moments=np.asarray(moment_record['expectations'])
    updates={};spin=dict(surface.spin_tensor_surfaces_MHz or {})
    for name,terms in fields.items():
        value=None
        for (active,contracted),coefficient in terms.items():
            polynomial=np.prod(points**np.asarray(active),axis=1)
            scalar=moments[:,:,words[contracted]]*polynomial[None,:]
            product=scalar[(...,)+(None,)*(coefficient.ndim-1)]*coefficient[:,None]
            value=product if value is None else value+product
        if value is None:raise ValueError('VSCF tensor expansions cannot be empty')
        value=value.reshape(surface.grid_shape+value.shape[2:])
        if name=='quadratic_kinetic_MHz':
            d=surface.dimension
            if value.shape!=surface.grid_shape+(d+3,d+3) or not np.allclose(value,value.swapaxes(-1,-2),atol=1e-10,rtol=1e-12):
                raise ValueError('joint VSCF kinetic tensor must be real symmetric with physical dimensions')
            f=value[...,:d,:d];mixed=value[...,:d,d:];rotation=value[...,d:,d:]
            if np.any(np.linalg.eigvalsh(f)<=0):
                raise ValueError('VSCF-contracted LAM kinetic form is not positive definite')
            rho=-np.linalg.solve(f,mixed).swapaxes(-1,-2)
            r=rotation-rho@f@rho.swapaxes(-1,-2);r=.5*(r+r.swapaxes(-1,-2))
            if np.any(np.linalg.eigvalsh(f)<=0) or np.any(np.linalg.eigvalsh(r)<=0):
                raise ValueError('VSCF-contracted quadratic kinetic form is not positive definite')
            updates.update(F_MHz=f,rho=rho,R_MHz=r)
        elif name.startswith('spin_tensor_surfaces_MHz.'):
            key=name.split('.',1)[1]
            if references[name]=='increment':
                if key not in spin:raise ValueError('an incremental spin surface requires its reference field')
                value=value+spin[key]
            spin[key]=value
        else:
            if references[name]=='increment':
                base=getattr(surface,name)
                if base is None:raise ValueError('an incremental tensor requires its reference field')
                value=value+base
            updates[name]=value
    if any(name.startswith('spin_tensor_surfaces_MHz.') for name in fields):updates['spin_tensor_surfaces_MHz']=spin
    return updates


def contract_lam_tabulated_vscf_fields(surface,specification,states,*,qualification_reports=None,
                                      require_qualified=True):
    """Contract the existing physical property expansions with tabulated modals.

    One GF TabulatedVSCFResult is supplied per native surface grid point, in
    C order. The potential is supplied by its physical producer; this function
    only averages property fields. Unqualified states require an explicit
    exploratory call and are labelled accordingly in surface metadata.
    """
    from dataclasses import replace
    metadata=dict(surface.metadata or {})
    if metadata.get('tabulated_vscf_tensor_contraction') or metadata.get('nmode_contraction',{}).get('contracted_tensor_fields'):
        raise ValueError('tensor fields have already been contracted')
    states=tuple(states);count=int(np.prod(surface.grid_shape))
    if len(states)!=count:raise ValueError('one tabulated VSCF state per surface point required')
    reports=None if qualification_reports is None else tuple(qualification_reports)
    if reports is not None and len(reports)!=count:
        raise ValueError('one numerical qualification report per surface point required')
    qualified=reports is not None and all(r.get('qualified') is True for r in reports)
    if require_qualified and not qualified:
        raise ValueError('qualified tabulated states required; exploratory contraction must be explicit')
    if reports is not None:
        for state,report in zip(states,reports,strict=True):
            last=report['history'][-1]
            if (not np.array_equal(state.domains,last['domains'])
                or any(len(q)!=last['mesh_checks'][-1]['points'] for q in state.coordinates)
                or not np.isclose(state.energy,last['mesh_checks'][-1]['energy'],rtol=0,atol=1e-10)):
                raise ValueError('tabulated VSCF states and qualification reports do not match')
    fields,references,words=prepare_vscf_tensor_fields(surface,specification)
    if not fields:raise ValueError('explicit tensor expansions required for a property contraction')
    active_dimension=surface.dimension-1
    points=(np.stack(np.meshgrid(*surface.coordinates[1:],indexing='ij'),axis=-1).reshape(-1,active_dimension)
        if active_dimension else np.empty((1,0)))
    moments=np.asarray([state.moments(words) for state in states]).reshape(surface.grid_shape[0],len(points),len(words))
    updates=contract_vscf_tensor_fields(surface,fields,references,points,dict(powers=words,expectations=moments))
    metadata=dict(surface.metadata or {})
    metadata['tabulated_vscf_tensor_contraction']=dict(schema='matrix.lock.tabulated_vscf_properties.v1',
        numerical_qualification_passed=qualified,fields=sorted(fields),
        scope='coordinate-property averages in supplied product VSCF states; potential remains the physical producer input')
    return replace(surface,metadata=metadata,**updates)
