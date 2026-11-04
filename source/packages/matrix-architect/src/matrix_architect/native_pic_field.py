"""Full native PIC E/G/H and reference anchoring without a harmonic tail.

Reference corrections are products of existing native coordinates, with every
support-bond switch. No Cartesian quadratic is evaluated away from reference.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
from scipy.linalg import lstsq
from matrix_chem import BOHR_TO_ANGSTROM
from matrix_chem.primitive_coordinates import eval_primitives, primitive_b_matrix
from matrix_chem.primitive_force_field import evaluate_primitive_functional_parameter, evaluate_torsion_fourier_parameter
from matrix_zaff.basic_primitive_derivative import analytic_basic_primitive_second_derivative
from .asymptotic_pic import AsymptoticPicFactor, AsymptoticPicCorrection, asymptotic_pic_bonded_energy_gradient
from .asymptotic_pic_hessian import NativeCorrectionHessianDesign


@dataclass(frozen=True)
class NativePicPoint:
    energy_hartree: float
    gradient_hartree_per_bohr: np.ndarray
    hessian_hartree_per_bohr2: np.ndarray


class NativePicField:
    """Resident full bonded native PES plus a caller-owned nonbonded E/G/H."""
    def __init__(self, primitives, field, nonbonded_evaluator, *, corrections=(), energy_offset_hartree=0., metadata=None):
        self.metadata=dict(metadata or {});self.native_zaff4_runtime=self.metadata.get("model_level")=="ZAFF4"
        self.schema='matrix.architect.native_pic_field.v1'
        self.active_zoom_level=self.metadata.get('model_level','NATIVE_PIC')
        from matrix_chem.topology.elements import atomic_symbol
        self.atoms=tuple(atomic_symbol(int(z)) for z in nonbonded_evaluator.atomic_numbers) if hasattr(nonbonded_evaluator,'atomic_numbers') else tuple(self.metadata.get('atoms',()))
        reference=self.metadata.get('reference_coordinates_angstrom')
        self.reference_coordinates_angstrom=None if reference is None else np.asarray(reference,float)
        self.primitives=tuple(primitives);self.field=field
        self.nonbonded_evaluator=nonbonded_evaluator
        self.corrections=tuple(corrections);self.energy_offset_hartree=float(energy_offset_hartree)
        self.functions={p.primitive_type_id:p for p in field.functional_parameters}
        self.torsions={p.primitive_type_id:p for p in field.torsion_fourier_parameters}
        terms=[tuple((i,'native',1,1) for i in t.primitive_indices) for t in field.retained_couplings]
        self.couplings=NativeCorrectionHessianDesign(self.primitives,field,terms)
        self.coupling_coefficients=np.array([t.force_constant for t in field.retained_couplings])
        terms=[tuple((f.primitive_index,f.form,f.power,f.harmonic,f.window_start,f.window_end) for f in c.factors) for c in self.corrections]
        self.correction_design=NativeCorrectionHessianDesign(self.primitives,field,terms)
        self.coefficients=np.array([c.coefficient for c in self.corrections])

    def energy_gradient(self,xyz):
        energy,gradient=asymptotic_pic_bonded_energy_gradient(self.field,self.primitives,xyz,corrections=self.corrections)
        nb=self.nonbonded_evaluator(xyz)
        return float(energy+nb.total_energy+self.energy_offset_hartree),gradient.reshape(-1)*BOHR_TO_ANGSTROM+nb.total_gradient.reshape(-1)

    def energy(self,xyz):
        return float(self.energies(np.asarray(xyz)[None,:,:])[0])

    def energies(self,geometries,*,batch_size=64):
        """Batch native energies, with identical mixed terms and bond switches."""
        xyz=np.asarray(geometries,float)
        if xyz.ndim!=3 or xyz.shape[1:]!=(len(self.nonbonded_evaluator.atomic_numbers) if hasattr(self.nonbonded_evaluator,'atomic_numbers') else 1+max(a for p in self.primitives for a in p.atoms),3) or not np.all(np.isfinite(xyz)):
            raise ValueError('finite Cartesian geometry batch required')
        if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size<1:raise ValueError('positive batch size required')
        if not hasattr(self,'_energy_terms'):
            factors=[tuple(AsymptoticPicFactor(i,'native') for i in t.primitive_indices) for t in self.field.retained_couplings]+[c.factors for c in self.corrections]
            coefficients=np.r_[self.coupling_coefficients,self.coefficients]
            unique=tuple(dict.fromkeys(factor for term in factors for factor in term));lookup={factor:i for i,factor in enumerate(unique)}
            width=max((len(t) for t in factors),default=0)
            indices=np.full((len(factors),width),len(unique),int)
            bonds=tuple(self.couplings.alpha);bond_position={b:i for i,b in enumerate(bonds)}
            support=np.zeros((len(factors),len(bonds)))
            for j,term in enumerate(factors):
                indices[j,:len(term)]=[lookup[t] for t in term]
                for b in set().union(*(self.couplings.supports[t.primitive_index] for t in term)):support[j,bond_position[b]]=1.
            self._energy_terms=(unique,indices,bonds,support,coefficients)
        unique,indices,bonds,support,coefficients=self._energy_terms
        answer=np.empty(len(xyz));q0=self.field.equilibrium_values
        for start in range(0,len(xyz),batch_size):
            block=xyz[start:start+batch_size]
            q=np.array([eval_primitives(self.primitives,x) for x in block]);delta=q-q0
            logs=-np.array([self.couplings.alpha[b]*delta[:,b] for b in bonds]).T**4
            values=np.ones((len(block),len(unique)+1))
            def native(i):
                if self.primitives[i].kind=='bond':return self.couplings.native_radial_jet(i,q[:,i])[0]
                if self.primitives[i].kind=='angle' and abs(np.sin(q0[i]))>1e-6:return (np.cos(q0[i])-np.cos(q[:,i]))/np.sin(q0[i])
                return np.sin(delta[:,i])
            for j,t in enumerate(unique):
                i=t.primitive_index;n=t.harmonic;z=delta[:,i]
                if t.form=='native':v=native(i)
                elif t.form=='inverse_distance':v=q0[i]**2*(1/q0[i]-1/q[:,i])
                elif t.form=='sin':v=np.sin(n*z)
                elif t.form=='one_minus_cos':v=1-np.cos(n*z)
                elif t.form=='jet_cos':v=np.cos(n*z)-1+n*n*(1-np.cos(z))
                elif t.form=='jet_sin':v=np.sin(n*z)-n*np.sin(z)
                elif t.form=='outer_window':
                    from .asymptotic_pic import periodic_outer_window
                    v=np.array([periodic_outer_window(x,t.window_start,t.window_end)[0] for x in z])
                else:raise ValueError('unsupported batch native factor: '+t.form)
                values[:,j]=v**t.power
            total=np.prod(values[:,indices],axis=2)*np.exp(logs@support.T)@coefficients
            for i,primitive in enumerate(self.primitives):
                parameter=(self.torsions if primitive.kind=='dihedral' else self.functions)[self.field.primitive_type_ids[i]]
                evaluate=evaluate_torsion_fourier_parameter if primitive.kind=='dihedral' else evaluate_primitive_functional_parameter
                local=np.array([evaluate(parameter,v)[0] for v in q[:,i]])
                if primitive.kind!='bond':
                    supported=[bonds.index(b) for b in self.couplings.supports[i]]
                    local*=np.exp(logs[:,supported].sum(axis=1))
                total+=local
            energy_only=getattr(self.nonbonded_evaluator,'energy',None)
            answer[start:start+len(block)]=total+np.array([
                energy_only(x) if energy_only is not None else self.nonbonded_evaluator(x).total_energy
                for x in block])+self.energy_offset_hartree
        return answer

    def __call__(self,xyz):
        xyz=np.asarray(xyz,float);p=self.primitives;f=self.field
        q=eval_primitives(p,xyz);b=primitive_b_matrix(p,xyz)
        gradient_q=np.zeros(len(p));hessian_q=np.zeros((len(p),len(p)))
        h=self.couplings.value(xyz)[2]@self.coupling_coefficients
        for i,primitive in enumerate(p):
            parameter=(self.torsions if primitive.kind=='dihedral' else self.functions)[f.primitive_type_ids[i]]
            v,d,dd=(evaluate_torsion_fourier_parameter if primitive.kind=='dihedral' else evaluate_primitive_functional_parameter)(parameter,q[i])
            dg=np.zeros(len(p));dg[i]=d
            dh=np.zeros_like(hessian_q);dh[i,i]=dd
            lg=np.zeros(len(p));lh=np.zeros(len(p));exponent=0.
            for j in (() if primitive.kind=='bond' else self.couplings.supports[i]):
                alpha=self.couplings.alpha[j];delta=q[j]-self.couplings.q0[j]
                exponent-=(alpha*delta)**4;lg[j]=-4*alpha**4*delta**3;lh[j]=-12*alpha**4*delta**2
            sw=np.exp(exponent)
            gradient_q+=sw*(dg+v*lg)
            hessian_q+=sw*(dh+np.outer(lg,dg)+np.outer(dg,lg)+v*(np.outer(lg,lg)+np.diag(lh)))
        h+=b.T@hessian_q@b
        for i,primitive in enumerate(p):
            item=SimpleNamespace(function={'bond':'R','angle':'A','dihedral':'D','out_of_plane':'U'}[primitive.kind],atoms=tuple(a+1 for a in primitive.atoms))
            for a,c,value in analytic_basic_primitive_second_derivative(item,xyz).entries:h[a,c]+=gradient_q[i]*value
        if self.corrections:h+=self.correction_design.value(xyz)[2]@self.coefficients
        e,g=asymptotic_pic_bonded_energy_gradient(f,p,xyz,corrections=self.corrections)
        nb=self.nonbonded_evaluator(xyz)
        return NativePicPoint(float(e+nb.total_energy+self.energy_offset_hartree),g.reshape(-1)*BOHR_TO_ANGSTROM+nb.total_gradient.reshape(-1),(h+h.T)/2*BOHR_TO_ANGSTROM**2+nb.total)


def anchor_native_pic_reference(runtime, reference_angstrom, target_hessian_bohr, *, target_gradient_bohr=None, relative_rank_cutoff=1e-10):
    """Fit reference G/H using switched native linear/quadratic corrections.

    The finite fit is a native reference match, not a global physical field
    certificate. Unresolved Cartesian residuals are reported and never hidden.
    """
    from itertools import combinations_with_replacement
    xyz=np.asarray(reference_angstrom,float);size=xyz.size
    target=np.asarray(target_hessian_bohr,float)
    gtarget=np.zeros(size) if target_gradient_bohr is None else np.asarray(target_gradient_bohr,float).reshape(-1)
    factors=[]
    for degree in (1,2):
        for indices in combinations_with_replacement(range(len(runtime.primitives)),degree):
            factors.append(tuple(AsymptoticPicFactor(i,'native',indices.count(i)) for i in sorted(set(indices))))
    design=NativeCorrectionHessianDesign(runtime.primitives,runtime.field,[tuple((t.primitive_index,t.form,t.power,t.harmonic) for t in row) for row in factors])
    _,dg,dh=design.value(xyz);base=runtime(xyz)
    triangle=np.triu_indices(size)
    # Fit in a single explicit angstrom convention. Cartesian rows include
    # geometric curvature, rather than a linear-only Wilson transformation.
    a=np.vstack([dg,dh[triangle]])
    y=np.r_[(gtarget-base.gradient_hartree_per_bohr)/BOHR_TO_ANGSTROM,(target-base.hessian_hartree_per_bohr2)[triangle]/BOHR_TO_ANGSTROM**2]
    coefficients,_,rank,_=lstsq(a,y,cond=relative_rank_cutoff,lapack_driver='gelsy')
    corrections=runtime.corrections+tuple(AsymptoticPicCorrection(float(v),f) for v,f in zip(coefficients,factors) if abs(v)>1e-14)
    anchored=NativePicField(runtime.primitives,runtime.field,runtime.nonbonded_evaluator,corrections=corrections,energy_offset_hartree=runtime.energy_offset_hartree)
    result=anchored(xyz)
    return anchored,dict(rank=int(rank),candidate_count=len(factors),gradient_max_abs_Eh_bohr=float(np.max(abs(result.gradient_hartree_per_bohr-gtarget))),hessian_max_abs_Eh_bohr2=float(np.max(abs(result.hessian_hartree_per_bohr2-target))),bare_quadratic_tail=False,physical_field_qualified=False)


@dataclass(frozen=True)
class FrozenNativeNonbonded:
    """Resident nonbonded terms with the exact fit-time topology and policy."""
    atomic_numbers: tuple
    bonds: tuple
    charges: np.ndarray
    parameters: dict
    charge_source: str

    def __call__(self,xyz):
        from matrix_zaff.nonbonded import architect_nonbonded_cartesian_hessian_components
        return architect_nonbonded_cartesian_hessian_components(np.asarray(xyz)/BOHR_TO_ANGSTROM,np.asarray(self.atomic_numbers),self.bonds,charges=self.charges,charge_source=self.charge_source,**self.parameters)

    def energy(self,xyz):
        from matrix_zaff.nonbonded import architect_nonbonded_cartesian_energy
        return architect_nonbonded_cartesian_energy(np.asarray(xyz)/BOHR_TO_ANGSTROM,
            np.asarray(self.atomic_numbers),self.bonds,charges=self.charges,
            charge_source=self.charge_source,**self.parameters)


def save_native_pic_field(path,runtime):
    """Portable JSON artifact containing the actual native functions and switches."""
    import json
    from pathlib import Path
    from dataclasses import fields,is_dataclass
    def encode(value):
        if isinstance(value,np.ndarray):return {'array':value.tolist(),'dtype':str(value.dtype)}
        if isinstance(value,np.generic):return encode(value.item())
        if isinstance(value,float) and not np.isfinite(value):return {'nonfinite':str(value)}
        if is_dataclass(value):return {'record':type(value).__name__,'fields':{f.name:encode(getattr(value,f.name)) for f in fields(value)}}
        if isinstance(value,tuple):return {'tuple':[encode(v) for v in value]}
        if isinstance(value,list):return [encode(v) for v in value]
        if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
        return value
    if not isinstance(runtime.nonbonded_evaluator,FrozenNativeNonbonded):raise TypeError('portable artifacts require a frozen native nonbonded record')
    payload=dict(schema='matrix.architect.native_pic_field.v1',primitives=encode(runtime.primitives),field=encode(runtime.field),nonbonded=encode(runtime.nonbonded_evaluator),corrections=encode(runtime.corrections),energy_offset_hartree=runtime.energy_offset_hartree,physical_field_qualified=False,metadata=encode(getattr(runtime,'metadata',{})))
    destination=Path(path)
    serialized=json.dumps(payload,allow_nan=False,separators=(',',':'))+'\n'
    with destination.open('x') as stream:stream.write(serialized)
    return destination


def load_native_pic_field(path):
    """Load only the explicitly supported native records; never execute pickle."""
    import json
    from pathlib import Path
    from matrix_chem.primitive_coordinates import Primitive
    from matrix_chem.primitive_force_field import PrimitiveMMForceField,AtomicMMType,PrimitiveMMType,PrimitiveMMCoupling,PrimitiveFunctionalParameter,TorsionFourierParameter,TorsionFourierTerm
    allowed={c.__name__:c for c in (Primitive,PrimitiveMMForceField,AtomicMMType,PrimitiveMMType,PrimitiveMMCoupling,PrimitiveFunctionalParameter,TorsionFourierParameter,TorsionFourierTerm,FrozenNativeNonbonded,AsymptoticPicCorrection,AsymptoticPicFactor)}
    def decode(value):
        if isinstance(value,list):return [decode(v) for v in value]
        if not isinstance(value,dict):return value
        if set(value)=={'nonfinite'}:
            if value['nonfinite'] not in ('inf','-inf','nan'):raise ValueError('invalid nonfinite metadata token')
            return float(value['nonfinite'])
        if set(value)=={'tuple'}:return tuple(decode(v) for v in value['tuple'])
        if set(value)=={'array','dtype'}:return np.asarray(value['array'],dtype=value['dtype'])
        if set(value)=={'record','fields'}:
            if value['record'] not in allowed:raise ValueError('unsupported native record')
            return allowed[value['record']](**{k:decode(v) for k,v in value['fields'].items()})
        return {k:decode(v) for k,v in value.items()}
    data=json.loads(Path(path).read_text())
    if data.get('schema')!='matrix.architect.native_pic_field.v1':raise ValueError('unsupported native PIC field schema')
    required={'schema','primitives','field','nonbonded','corrections','energy_offset_hartree','physical_field_qualified','metadata'}
    if set(data)!=required:raise ValueError('incomplete or unknown native PIC field entries')
    if data['physical_field_qualified'] is not False:raise ValueError('a portable native candidate is not an independently qualified field')
    if not np.isfinite(data['energy_offset_hartree']):raise ValueError('finite native energy offset required')
    return NativePicField(decode(data['primitives']),decode(data['field']),decode(data['nonbonded']),corrections=decode(data['corrections']),energy_offset_hartree=data['energy_offset_hartree'],metadata=decode(data.get('metadata',{})))
