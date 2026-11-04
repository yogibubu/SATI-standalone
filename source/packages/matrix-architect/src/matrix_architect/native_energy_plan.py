"""Compiled energy-only plans for unchanged native PIC/ZAFF potentials.

Plans freeze algebra and metadata, never a geometry-dependent topology. Every
geometry recomputes primitive values, support switches and all nonbonded paths.
Unsupported atlases/policies are rejected rather than silently approximated.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def _norm(v):
    return math.sqrt(np.dot(v,v))


@njit(cache=True)
def _coordinates(xyz,kinds,atoms):
    q=np.empty(len(kinds))
    for p in range(len(kinds)):
        i,j,k,l=atoms[p];kind=kinds[p]
        if kind==1:q[p]=_norm(xyz[i]-xyz[j])
        elif kind==2:
            u=xyz[i]-xyz[j];v=xyz[k]-xyz[j];a=_norm(u);b=_norm(v)
            cosine=0. if a<1e-12 or b<1e-12 else np.dot(u/a,v/b)
            q[p]=math.acos(min(1.,max(-1.,cosine)))
        elif kind==3:
            a=xyz[i]-xyz[j];b=xyz[k]-xyz[j];c=xyz[l]-xyz[k]
            n1=np.cross(a,b);n2=np.cross(b,c);norm=_norm(b)
            unit=np.zeros(3) if norm<1e-12 else b/norm
            q[p]=math.atan2(np.dot(np.cross(n1,n2),unit),np.dot(n1,n2))
        else:
            v=xyz[l]-xyz[i];n=np.cross(xyz[j]-xyz[i],xyz[k]-xyz[i]);den=_norm(v)*_norm(n)
            q[p]=0. if den<1e-12 else -math.asin(min(1.,max(-1.,np.dot(v,n)/den)))
    return q


@njit(cache=True)
def _terms(values,indices,coefficients,groups,switches):
    total=0.
    for term in range(len(coefficients)):
        product=1.
        for position in range(indices.shape[1]):product*=values[indices[term,position]]
        total+=coefficients[term]*switches[groups[term]]*product
    return total


@njit(cache=True)
def _nonbonded(xyz,pairs,pair_parameters,settings):
    # Inputs use the reference owner's bohr conversion and radial constants.
    unit,alpha,cutoff,scale14,vdw,emodel,model=settings
    x=xyz/unit;n=len(x);distance=np.zeros((n,n));edge=np.zeros((n,n))
    for p in range(len(pairs)):
        i,j=pairs[p];r=_norm(x[i]-x[j])
        if r<=1e-12:raise ValueError('coincident nonbonded atoms')
        distance[i,j]=r;distance[j,i]=r
        radius=pair_parameters[p,0]
        if radius>0 and r<=cutoff:
            e=.5*(1.+math.erf(alpha*(radius-r)));edge[i,j]=e;edge[j,i]=e
    ee=0.;ev=0.
    for p in range(len(pairs)):
        i,j=pairs[p];r=distance[i,j]
        scale=1.
        if vdw or emodel==2:
            product2=1.
            for k in range(n):
                if k!=i and k!=j:product2*=1.-edge[i,k]*edge[k,j]
            union2=1.-product2;union3=0.
            if scale14!=1.:
                product3=1.
                for k in range(n):
                    if k==i or k==j:continue
                    for l in range(n):
                        if l==i or l==j or l==k:continue
                        product3*=1.-edge[i,k]*edge[k,l]*edge[l,j]
                union3=1.-product3
            scale=(1.-edge[i,j])*(1.-union2)*(1.-(1.-scale14)*union3)
        charge=pair_parameters[p,1]
        if abs(charge)>1e-16:
            if emodel==1:ee+=charge*math.erf(pair_parameters[p,2]*r)/r
            elif emodel==2:ee+=scale*charge/r
        if vdw:
            rmin=pair_parameters[p,3];epsilon=pair_parameters[p,4];ratio=r/rmin
            if model==1:
                a=math.sqrt(160.);polynomial=ratio**4-2.*ratio**2+3.
                value=math.exp(a*(1.-ratio))-polynomial*math.exp(.5*a*(1.-ratio))
            else:
                power=(rmin/r)**6;value=power*power-2.*power
            ev+=scale*epsilon*value
    return ee+ev


@njit(cache=True,nogil=True)
def _energies(xyz,kinds,atoms,q0,alpha,bonds,supports,forms,parameters,
              torsion_primitive,torsion_order,torsion_cosine,torsion_sine,
              factor_data,base_indices,base_coefficients,base_groups,
              full_indices,full_coefficients,full_groups,
              completed,scales,radius,offset,pairs,pair_parameters,nb_settings):
    result=np.empty(len(xyz))
    for point in range(len(xyz)):
        q=_coordinates(xyz[point],kinds,atoms);delta=q-q0
        logs=np.zeros(len(q))
        for b in bonds:logs[b]=-(alpha[b]*delta[b])**4
        switches=np.empty(len(supports))
        for group in range(len(supports)):
            exponent=0.
            for b in range(len(q)):
                if supports[group,b]:exponent+=logs[b]
            switches[group]=math.exp(exponent)
        weight=1.
        if completed:
            s=0.
            for i in range(len(q)):
                if kinds[i]==1:
                    value=q0[i]**2*(1./q0[i]-1./q[i])/scales[i];s+=value*value
                elif kinds[i]==2:
                    value=(math.cos(q0[i])-math.cos(q[i]))/math.sin(q0[i])/scales[i];s+=value*value
                else:s+=4.*math.sin(delta[i]/2.)**2/scales[i]**2
            a=s/radius**2
            weight=0. if a>700.**(1./3.) else math.exp(-a**3)
        values=np.ones(len(factor_data)+1)
        for f in range(len(factor_data)):
            i=int(factor_data[f,0]);form=int(factor_data[f,1]);power=int(factor_data[f,2]);n=factor_data[f,3];z=delta[i]
            if form==0:
                if kinds[i]==1:v=-math.expm1(-alpha[i]*z)/alpha[i]
                elif kinds[i]==2 and abs(math.sin(q0[i]))>1e-6:v=(math.cos(q0[i])-math.cos(q[i]))/math.sin(q0[i])
                else:v=math.sin(z)
            elif form==1:v=q0[i]**2*(1./q0[i]-1./q[i])
            elif form==2:v=math.sin(n*z)
            elif form==3:v=1.-math.cos(n*z)
            elif form==4:v=math.cos(n*z)-1.+n*n*(1.-math.cos(z))
            elif form==5:v=math.sin(n*z)-n*math.sin(z)
            else:
                start=factor_data[f,4];end=factor_data[f,5];d=abs(math.atan2(math.sin(z),math.cos(z)))
                if d<=start+8.*np.finfo(np.float64).eps:v=0.
                elif d>=end-8.*np.finfo(np.float64).eps:v=1.
                else:
                    t=(d-start)/(end-start);v=t**3*(10.+t*(-15.+6.*t))
            values[f]=v**power
        one_body=0.
        for i in range(len(q)):
            p=parameters[i];form=forms[i];v=0.
            if form==0:v=p[0]  # Fourier constant; terms below
            elif form==1:v=.5*p[1]*(q[i]-p[0])**2
            elif form==2:v=p[2]*(1.-math.exp(-p[1]*(q[i]-p[0])))**2
            elif form==3:v=p[1]*(1./q[i]-1./p[0])**2
            elif form==4:v=p[2]*(1.-(p[0]/q[i])**p[1])**2
            elif form==5:v=p[1]*(math.cos(q[i])-math.cos(p[0]))**2
            else:
                u=math.cos(q[i])-math.cos(p[0]) if form==6 else p[0]**2*(1./p[0]-1./q[i])
                v=p[5]
                for order in range(3,-1,-1):v=p[order+1]+u*v
                if form==7:
                    t=p[7]*(q[i]-p[0]);v=p[6] if abs(t)>4. else p[6]+math.exp(-t**6)*(v-p[6])
            if kinds[i]!=1:v*=switches[int(p[9])]
            one_body+=v
        for t in range(len(torsion_primitive)):
            i=torsion_primitive[t];order=torsion_order[t]
            one_body+=(torsion_cosine[t]*math.cos(order*q[i])+torsion_sine[t]*math.sin(order*q[i]))*switches[int(parameters[i,9])]
        common=one_body+_nonbonded(xyz[point],pairs,pair_parameters,nb_settings)+offset
        base=_terms(values,base_indices,base_coefficients,base_groups,switches)+common
        if completed and weight>0.:
            full=_terms(values,full_indices,full_coefficients,full_groups,switches)+common
            result[point]=base+weight*(full-base)
        else:result[point]=base
    return result


class CompiledNativeEnergyPlan:
    """Frozen energy plan with a batch API compatible with NativePicField.

    Construction may raise NotImplementedError for unsupported primitives or
    nonbonded policies. Callers must then use the public reference evaluator.
    No fast-math reassociation or geometry-dependent cutoff is introduced.
    """
    def __init__(self,runtime):
        from .native_pic_field import NativePicField,FrozenNativeNonbonded
        from .native_coupling_completion import NativeCouplingCompletedField
        from .asymptotic_pic import AsymptoticPicFactor
        from matrix_chem import BOHR_TO_ANGSTROM
        from matrix_chem.nonbonded import uff_pair_parameters
        from matrix_chem.continuous_nonbonded import BOHR_TO_ANGSTROM as TOPOLOGY_BOHR
        from matrix_chem.topology.covalent_radii import covalent_radius
        from matrix_chem.topology.descriptor_parameters import CNA_SCALE
        self.runtime=runtime;completed=isinstance(runtime,NativeCouplingCompletedField)
        fitted=runtime.fitted if completed else runtime
        baseline=runtime.baseline if completed else runtime
        if not isinstance(fitted,NativePicField) or not isinstance(fitted.nonbonded_evaluator,FrozenNativeNonbonded):
            raise NotImplementedError('resident native field with frozen nonbonded required')
        kinds={'bond':1,'angle':2,'dihedral':3,'out_of_plane':4}
        if any(p.kind not in kinds for p in fitted.primitives):raise NotImplementedError('unsupported primitive atlas')
        nb=fitted.nonbonded_evaluator;settings=dict(nb.parameters)
        allowed={'electrostatic','uff_vdw','vdw_model','one_four_scale','vdw_radius_scale','vdw_well_depth_scale',
            'vdw_damping_scale','vdw_damping_exponent','electrostatic_model','electrostatic_gaussian_width_scale',
            'topology_scaling','topology_switch_alpha_per_angstrom'}
        if set(settings)-allowed or settings.get('topology_scaling','continuous_erf_paths')!='continuous_erf_paths':
            raise NotImplementedError('unsupported nonbonded settings')
        if settings.get('vdw_model','uff_12_6') not in ('uff_12_6','exppe_from_uff'):
            raise NotImplementedError('unsupported vdW model')
        if settings.get('electrostatic_model','topology_scaled_point_charges') not in ('gaussian_erf_all_pairs','topology_scaled_point_charges'):
            raise NotImplementedError('unsupported electrostatic model')
        self.natoms=len(nb.atomic_numbers);pcount=len(fitted.primitives)
        pkinds=np.array([kinds[p.kind] for p in fitted.primitives]);atoms=np.full((pcount,4),-1,dtype=int)
        for i,p in enumerate(fitted.primitives):atoms[i,:len(p.atoms)]=p.atoms
        q0=np.asarray(fitted.field.equilibrium_values,float);alpha=np.zeros(pcount)
        bonds=np.array(tuple(fitted.couplings.alpha),dtype=int)
        for b in bonds:alpha[b]=fitted.couplings.alpha[b]
        support_lookup={};support_rows=[]
        def support_id(primitives):
            support=tuple(sorted(set().union(*(fitted.couplings.supports[i] for i in primitives)))) if primitives else ()
            if support not in support_lookup:
                support_lookup[support]=len(support_rows);row=np.zeros(pcount,dtype=np.bool_);row[list(support)]=True;support_rows.append(row)
            return support_lookup[support]
        forms=np.empty(pcount,dtype=int);parameters=np.zeros((pcount,10));tp=[];to=[];tc=[];ts=[]
        for i,primitive in enumerate(fitted.primitives):
            parameters[i,9]=support_id([] if primitive.kind=='bond' else [i])
            if primitive.kind=='dihedral':
                parameter=fitted.torsions[fitted.field.primitive_type_ids[i]];forms[i]=0;parameters[i,0]=parameter.constant
                for term in parameter.terms:
                    tp.append(i);to.append(int(term.periodicity));tc.append(term.cosine_coefficient);ts.append(term.sine_coefficient)
                continue
            parameter=fitted.functions[fitted.field.primitive_type_ids[i]];p=dict(parameter.parameters);form=parameter.form
            if form in ('harmonic','harmonic_fallback','linear_bend_component'):
                forms[i]=1;parameters[i,:2]=[p.get('R0',p.get('THETA0',p.get('Q0'))),p['K']]
            elif form=='morse':forms[i]=2;parameters[i,:3]=[p['R0'],p['ALPHA'],p['D']]
            elif form=='inverse_distance':forms[i]=3;parameters[i,:2]=[p['R0'],p['A']]
            elif form=='inverse_power_stretch':forms[i]=4;parameters[i,:3]=[p['R0'],p['P'],p['D']]
            elif form=='cosine':forms[i]=5;parameters[i,:2]=[p['THETA0'],p['A']]
            elif form in ('cosine_quartic','inverse_distance_quartic_switched'):
                forms[i]=6 if form=='cosine_quartic' else 7;parameters[i,0]=p['THETA0' if forms[i]==6 else 'R0']
                parameters[i,1:6]=[p['C'+str(k)] for k in range(5)]
                if forms[i]==7:parameters[i,6:8]=[p['E_INFINITY'],p['ALPHA']]
            else:raise NotImplementedError('unsupported one-dimensional native function: '+form)
        all_factors=[];factor_lookup={};factor_codes={'native':0,'inverse_distance':1,'sin':2,'one_minus_cos':3,'jet_cos':4,'jet_sin':5,'outer_window':6}
        def pack_terms(field):
            terms=[(tuple(AsymptoticPicFactor(i,'native') for i in t.primitive_indices),t.force_constant) for t in field.field.retained_couplings]
            terms.extend((c.factors,c.coefficient) for c in field.corrections)
            width=max((len(term) for term,_ in terms),default=0);ids=[];coefs=[];groups=[]
            for term,coefficient in terms:
                row=[]
                for factor in term:
                    if factor.form not in factor_codes:raise NotImplementedError('unsupported native product factor')
                    if factor not in factor_lookup:
                        factor_lookup[factor]=len(all_factors)
                        all_factors.append([factor.primitive_index,factor_codes[factor.form],factor.power,factor.harmonic,
                            factor.window_start or 0.,factor.window_end or 0.])
                    row.append(factor_lookup[factor])
                ids.append(row);coefs.append(coefficient);groups.append(support_id([f.primitive_index for f in term]))
            return ids,width,np.asarray(coefs),np.asarray(groups,dtype=int)
        base=pack_terms(baseline);full=pack_terms(fitted) if completed else base
        def term_arrays(packed):
            ids,width,coef,groups=packed;indices=np.full((len(ids),width),len(all_factors),dtype=int)
            for i,row in enumerate(ids):indices[i,:len(row)]=row
            return indices,coef,groups
        base_args=term_arrays(base);full_args=term_arrays(full)
        pairs=np.array([(i,j) for i in range(self.natoms) for j in range(i+1,self.natoms)],dtype=int).reshape(-1,2)
        pp=np.zeros((len(pairs),5));radii=[covalent_radius(int(z)) for z in nb.atomic_numbers]
        switch_alpha=float(settings.get('topology_switch_alpha_per_angstrom',8.))*TOPOLOGY_BOHR
        valid=[r for r in radii if r is not None]
        cutoff=2*CNA_SCALE*max(valid)/TOPOLOGY_BOHR+8/switch_alpha if valid else 0.
        electrostatic=bool(settings.get('electrostatic',False));vdw=bool(settings.get('uff_vdw',False))
        if (float(settings.get('one_four_scale',.5))<0
                or float(settings.get('vdw_radius_scale',1.))<=0
                or float(settings.get('vdw_well_depth_scale',1.))<=0
                or float(settings.get('electrostatic_gaussian_width_scale',1.))<=0):
            raise ValueError('invalid frozen pair scales')
        emodel=0 if not electrostatic else 1 if settings.get('electrostatic_model')=='gaussian_erf_all_pairs' else 2
        widths=np.array([float(settings.get('electrostatic_gaussian_width_scale',1.))*(float(r) if r is not None else .75)/BOHR_TO_ANGSTROM for r in radii])
        for p,(i,j) in enumerate(pairs):
            pp[p,0]=CNA_SCALE*(radii[i]+radii[j])/TOPOLOGY_BOHR if radii[i] is not None and radii[j] is not None else 0.
            if electrostatic:pp[p,1:3]=[nb.charges[i]*nb.charges[j],1/np.sqrt(2*(widths[i]**2+widths[j]**2))]
            if vdw:
                rmin,epsilon=uff_pair_parameters(nb.atomic_numbers[i],nb.atomic_numbers[j])
                pp[p,3:5]=[rmin*float(settings.get('vdw_radius_scale',1.)),epsilon*float(settings.get('vdw_well_depth_scale',1.))]
        nbs=np.array([BOHR_TO_ANGSTROM,switch_alpha,cutoff,float(settings.get('one_four_scale',.5)),
            float(vdw),float(emodel),1. if settings.get('vdw_model')=='exppe_from_uff' else 0.])
        if switch_alpha<=0 or not np.all(np.isfinite(nbs)) or not np.all(np.isfinite(pp)):
            raise ValueError('invalid frozen nonbonded parameters')
        scales=runtime.scales if completed else np.ones(pcount);radius=runtime.radius if completed else 1.
        if completed and any(p.kind=='angle' and abs(np.sin(q0[i]))<1e-6 for i,p in enumerate(fitted.primitives)):
            raise NotImplementedError('singular completion angular chart')
        self.arguments=(pkinds,atoms,q0,alpha,bonds,np.array(support_rows),forms,parameters,
            np.array(tp,dtype=int),np.array(to,dtype=int),np.array(tc),np.array(ts),np.asarray(all_factors).reshape(-1,6),
            *base_args,*full_args,completed,np.asarray(scales),float(radius),float(fitted.energy_offset_hartree),pairs,pp,nbs)
        self.diagnostics=dict(implementation='compiled native energy plan',primitive_count=pcount,
            factor_count=len(all_factors),base_terms=len(base[0]),full_terms=len(full[0]),support_groups=len(support_rows),
            potential_modified=False,fast_math=False,geometry_dependent_topology_recomputed=True)

    def energies(self,geometries,*,batch_size=128,workers=1):
        xyz=np.asarray(geometries,float)
        if xyz.ndim!=3 or xyz.shape[1:]!=(self.natoms,3) or not np.all(np.isfinite(xyz)):
            raise ValueError('finite Cartesian geometry batch required')
        if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size<1:raise ValueError('positive batch size required')
        if not isinstance(workers,int) or isinstance(workers,bool) or workers<1:
            raise ValueError('positive worker count required')
        if workers==1 or len(xyz)<2*workers:return _energies(xyz,*self.arguments)
        # The compiled batch releases the GIL. Geometry slices share immutable
        # plan arrays, preserve point order and perform identical point algebra.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts=list(pool.map(lambda part:_energies(part,*self.arguments),
                                np.array_split(xyz,workers)))
        return np.concatenate(parts)

    def energy(self,geometry):return float(self.energies(np.asarray(geometry)[None])[0])
