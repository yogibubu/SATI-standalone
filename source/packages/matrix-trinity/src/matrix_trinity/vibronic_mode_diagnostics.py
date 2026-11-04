"""Mode-motion and harmonic wavepacket diagnostics, without rate attribution."""
import numpy as np
from matrix_gf.harmonic import CM_PER_HARTREE,HESSIAN_EIGENVALUE_TO_CM


def normal_mode_motion_diagnostics(coordinates_bohr,masses_amu,mass_weighted_modes,
                                   frequencies_cm,bonds):
    """Describe Cartesian motion and internal-coordinate derivatives per q.

    Bond/angle derivative rankings are geometric descriptors, not a potential
    energy distribution and not a mode-by-mode rate decomposition.
    """
    xyz=np.asarray(coordinates_bohr,float);m=np.asarray(masses_amu,float)
    modes=np.asarray(mass_weighted_modes,float).reshape(-1,len(m),3)
    w=np.asarray(frequencies_cm,float);edges=np.asarray(bonds)
    if (xyz.shape!=(len(m),3) or len(w)!=len(modes) or np.any(w<=0) or np.any(m<=0)
            or edges.ndim!=2 or edges.shape[1]!=2 or edges.dtype.kind not in 'iu'
            or np.any(edges<0) or np.any(edges>=len(m)) or np.any(edges[:,0]==edges[:,1])
            or not all(np.all(np.isfinite(a)) for a in (xyz,m,modes,w))):
        raise ValueError('finite geometry, positive modes/masses and valid bond indices required')
    if not np.allclose(modes.reshape(len(w),-1)@modes.reshape(len(w),-1).T,np.eye(len(w)),atol=1e-6,rtol=1e-6):
        raise ValueError('mass-weighted orthonormal mode rows required')
    centered=xyz-np.average(xyz,axis=0,weights=m)
    normal=np.linalg.svd(centered*np.sqrt(m[:,None]))[2][-1]
    dx=modes/np.sqrt(m)[None,:,None]*(HESSIAN_EIGENVALUE_TO_CM/np.sqrt(CM_PER_HARTREE*w))[:,None,None]
    neighbors={i:[] for i in range(len(m))}
    for i,j in edges:neighbors[int(i)].append(int(j));neighbors[int(j)].append(int(i))
    records=[]
    for k,displacement in enumerate(dx):
        stretches=[];angles=[]
        for i,j in edges:
            bond=xyz[j]-xyz[i];length=np.linalg.norm(bond)
            stretches.append(dict(atoms_1based=[int(i+1),int(j+1)],derivative_bohr_per_q=float(bond@(displacement[j]-displacement[i])/length)))
        for j,adjacent in neighbors.items():
            for ii,i in enumerate(adjacent):
                for l in adjacent[:ii]:
                    a,b=xyz[i]-xyz[j],xyz[l]-xyz[j];la,lb=np.linalg.norm(a),np.linalg.norm(b)
                    cosine=float(a@b/la/lb);sine=np.sqrt(max(0.,1-cosine*cosine))
                    if sine<1e-8:continue
                    da,db=displacement[i]-displacement[j],displacement[l]-displacement[j]
                    dc=(da@b+a@db)/la/lb-cosine*(a@da/la**2+b@db/lb**2)
                    angles.append(dict(atoms_1based=[i+1,j+1,l+1],derivative_radian_per_q=float(-dc/sine)))
        records.append(dict(mode_1based=k+1,harmonic_frequency_cm=float(w[k]),
            out_of_plane_mass_weighted_fraction=float(np.sum((modes[k]@normal)**2)),
            atomic_mass_weighted_participation=np.sum(modes[k]**2,axis=1).tolist(),
            largest_bond_derivatives=sorted(stretches,key=lambda row:-abs(row['derivative_bohr_per_q']))[:4],
            largest_angle_derivatives=sorted(angles,key=lambda row:-abs(row['derivative_radian_per_q']))[:4]))
    return dict(schema='matrix.trinity.mode_motion_diagnostics.v1',modes=records,
        interpretation='Geometric mode descriptors, not force-weighted PED or rate contributions')


def harmonic_final_occupation_moments(generator,mode_count):
    """Mean final occupation for the normalized harmonic initial vacuum.

    The generating function orders initial variables before final variables.
    These integrated wavepacket moments include Duschinsky/squeezing but no
    energy-conservation filter and no anharmonic initial-state correction.
    """
    n=int(mode_count);matrix=np.asarray(generator['matrix'],float);linear=np.asarray(generator['linear'],float)
    if matrix.shape!=(2*n,2*n) or linear.shape!=(2*n,) or n<=0:raise ValueError('Gaussian dimension mismatch')
    a=matrix[n:,n:];b=linear[n:];identity=np.eye(n)
    if max(abs(np.linalg.eigvalsh(a)))>=1:raise ValueError('non-normalizable final Gaussian')
    vx=.5*np.linalg.solve(identity-a,identity+a);vp=.5*np.linalg.solve(identity+a,identity-a)
    mean=np.sqrt(2)*np.linalg.solve(identity-a,b)
    occupation=(np.diag(vx)+np.diag(vp)+mean**2-1)/2
    if np.min(occupation)<-1e-10:raise ArithmeticError('negative mean occupation')
    return dict(mean_occupation=occupation.tolist(),mean_dimensionless_position=mean.tolist(),
        energy_filter_applied=False,anharmonic_initial_state_used=False,
        role='integrated harmonic wavepacket diagnostic, not a golden-rule rate contribution')
