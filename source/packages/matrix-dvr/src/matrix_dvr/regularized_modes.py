"""Reuse resident DVR regularizers for high-occupation transverse mode tables.

Potential completion and occupation-domain certification are distinct. A table
can replace diagonal HDC extrapolation; residual cross terms must still pass
the occupation-domain certificate. The absolute ZPE is supplied independently.
"""
import numpy as np
from scipy.linalg import eigh
from matrix_gf.regularization import (handy_gaussian_derivative_potential,
    handy_morse_derivative_potential,RegularizedQuarticPotential)


def regularized_transverse_potential(q,f2,f3,f4,*,model='handy-gaussian',beta=None,
                                      damping_radius=None,wall_scale=6.,wall_height_cm=10000.,wall_degree=8):
    """Preserve the local F2/F3/F4 jet and explicitly record the tail model.

    Handy Gaussian/Morse are reused verbatim from the resident DVR. The
    protected super-Gaussian uses ARCHITECT's exp[-(s/R^2)^3] completion rule,
    specialized to one dimension and a harmonic frozen parent. It changes
    the cubic correction first at order nine. No universal tail is inferred.
    """
    q=np.asarray(q,float)
    if (not np.all(np.isfinite(q)) or not np.all(np.isfinite([f2,f3,f4,wall_scale,wall_height_cm]))
            or f2<=0 or wall_scale<=0 or wall_height_cm<0 or wall_degree<6 or wall_degree%2):
        raise ValueError('stable local mode, finite parameters and even wall degree >=6 required')
    if model=='quartic-single-well':
        # V'(q)/q=F2+F3*q/2+F4*q^2/6. Its positive minimum
        # proves that the origin is the only stationary point. A stable
        # hardening quartic needs no remote damping or artificial wall.
        if f4<0 or (f4==0 and f3!=0):
            raise ValueError('globally single-well confining quartic required')
        margin=f2 if f4==0 else f2-3*f3*f3/(8*f4)
        if margin<=0:raise ValueError('quartic has no strict single-well certificate')
        potential=f2*q*q/2+f3*q**3/6+f4*q**4/24
        metadata=dict(model=model,mathematically_confining=True,
            unique_minimum_proved=True,minimum_radial_stiffness_cm=float(margin),
            remote_damping_applied=False)
    elif model=='handy-gaussian':
        if f3!=0:raise ValueError('Handy Gaussian requires zero cubic; use protected-gaussian for a general jet')
        beta=(abs(f4)/(6*f2) if f4 else 1/wall_scale**2) if beta is None else beta
        potential,metadata=handy_gaussian_derivative_potential(q,f2,f4,beta)
    elif model=='handy-morse':
        potential,metadata=handy_morse_derivative_potential(q,f2,f3,f4)
        if metadata['handy_morse_y4_coeff_cm-1']< -1e-10:
            raise ValueError('Handy Morse has a negative leading tail; use protected completion or a physical potential')
    elif model=='protected-gaussian':
        if damping_radius is None or not np.isfinite(damping_radius) or damping_radius<=0:
            raise ValueError('explicit positive damping radius required')
        completed=RegularizedQuarticPotential(np.array([[f2]]),np.full((1,)*3,f3),
            np.full((1,)*4,f4),damping_radius)
        potential=np.array([completed.energy([x]) for x in q.flat]).reshape(q.shape)
        metadata=dict(form='harmonic + exp[-(q/R)^6]*(cubic+quartic)',damping_radius=damping_radius,
                      native_rule='matrix_architect.native_coupling_completion.NativeCouplingCompletedField',
                      leading_added_cubic_order=9)
    else:raise ValueError('unknown resident transverse regularization model')
    potential=potential+wall_height_cm*(q/wall_scale)**wall_degree
    metadata.update(model=model,shared_implementation=('matrix_dvr.regularized_modes' if model=='quartic-single-well' else 'matrix_gf.regularization'),wall_scale=wall_scale,
                    wall_height_cm=wall_height_cm,wall_degree=wall_degree,local_jet_preserved_through=4)
    return potential,metadata


def completed_diagonal_metric(q,frequency_cm,first_cm,second_cm,*,damping_radius=None):
    """Positive diagonal inverse mass preserving G/G1/G2 at the origin.

    Retain a globally positive quadratic jet exactly. Otherwise reuse the
    protected exp[-(q/R)^6] completion and certify a global lower bound.
    """
    q=np.asarray(q,float);w,a,b=map(float,(frequency_cm,first_cm,second_cm))
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite([w,a,b])) or w<=0:
        raise ValueError('finite positive diagonal reference metric required')
    margin=w-a*a/(2*b) if b>0 else (w if a==0 and b==0 else -np.inf)
    if margin>0:
        return w+a*q+b*q*q/2,dict(form='quadratic',positive_lower_bound_cm=margin,metric_jet_preserved_through=2)
    if damping_radius is None or not np.isfinite(damping_radius) or damping_radius<=0:
        raise ValueError('quadratic diagonal metric is not globally positive; explicit protected radius required')
    R=float(damping_radius)
    peak1=R*(1/6)**(1/6)*np.exp(-1/6);peak2=R*R*(1/3)**(1/3)*np.exp(-1/3)
    lower=w-abs(a)*peak1-max(0.,-b)*peak2/2
    if lower<=0:raise ValueError('protected diagonal metric has no positive global certificate')
    return w+np.exp(-(q/R)**6)*(a*q+b*q*q/2),dict(form='protected-gaussian',
        damping_radius=R,positive_lower_bound_cm=lower,metric_jet_preserved_through=2)


def regularized_mode_levels(frequency_cm,f3_cm,f4_cm,*,max_quanta,grid_points=240,
                              half_width=8.,fundamental_anchor_cm=None,
                              metric_first_cm=0.,metric_second_cm=0.,metric_damping_radius=None,**potential_controls):
    """Sine DVR on a fixed regularized potential, returning excitation levels.

    A declared fundamental anchor adds a linear occupation correction to the
    DVR excitations. It preserves the HDC fundamental without recreating a
    quadratic descending diagonal tail; monotonicity is checked afterwards.
    Grid width and wall scale are independent to allow genuine box convergence.
    """
    if (isinstance(max_quanta,bool) or not isinstance(max_quanta,int) or max_quanta<1
            or not isinstance(grid_points,int) or grid_points<2*max_quanta+8
            or not np.isfinite(half_width) or half_width<=0):
        raise ValueError('positive quanta, sufficiently large grid and finite box required')
    q=-half_width+2*half_width/(grid_points+1)*np.arange(1,grid_points+1)
    potential,info=regularized_transverse_potential(q,frequency_cm,f3_cm,f4_cm,**potential_controls)
    # Exact finite sine-grid kinetic operator for omega*p^2/2.
    index=np.arange(1,grid_points+1)
    basis=np.sqrt(2/(grid_points+1))*np.sin(np.pi*np.outer(index,index)/(grid_points+1))
    kinetic=frequency_cm/2*(np.pi*index/(2*half_width))**2
    if not np.all(np.isfinite([metric_first_cm,metric_second_cm])):
        raise ValueError('finite diagonal metric derivatives required')
    if metric_first_cm or metric_second_cm:
        # Hkin=p[omega+G1*q+G2*q^2/2]p/2. Use exact sine-Galerkin
        # cosine moments, including the p q p and p q q p ordering.
        _,metric_info=completed_diagonal_metric(np.array([0.]),frequency_cm,metric_first_cm,metric_second_cm,
            damping_radius=metric_damping_radius)
        if metric_info['form']=='protected-gaussian':
            from scipy.special import roots_legendre
            nodes,weights=roots_legendre(max(800,2*grid_points));nodes*=half_width;weights*=half_width
            metric,_=completed_diagonal_metric(nodes,frequency_cm,metric_first_cm,metric_second_cm,
                damping_radius=metric_damping_radius)
            momenta=np.pi*index/(2*half_width)
            derivatives=np.cos((nodes[:,None]+half_width)*momenta)*momenta/np.sqrt(half_width)
            T=np.diag(kinetic)+(derivatives.T*(weights*(metric-frequency_cm)/2))@derivatives
            hamiltonian=basis@T@basis.T+np.diag(potential)
            info.update(diagonal_metric=metric_info,kinetic_ordering='p G(q) p / 2')
        else:
            hamiltonian=None
    else:hamiltonian=(basis*kinetic)@basis.T+np.diag(potential)
    if (metric_first_cm or metric_second_cm) and hamiltonian is None:
        ell=np.arange(2*grid_points+1,dtype=float);z=ell[1:]
        parity=(-1.)**z;I=np.zeros(len(ell));J=np.zeros(len(ell));J[0]=2*half_width**3/3
        I[1:]=(parity-1)*(2*half_width/(np.pi*z))**2
        J[1:]=2*half_width*(parity+1)*(2*half_width/(np.pi*z))**2
        difference=abs(index[:,None]-index[None,:]);total=index[:,None]+index[None,:]
        cq=(I[difference]+I[total])/(2*half_width)
        cq2=(J[difference]+J[total])/(2*half_width)
        momenta=np.pi*index/(2*half_width)
        T=np.diag(kinetic)+np.outer(momenta,momenta)*(metric_first_cm*cq/2+metric_second_cm*cq2/4)
        hamiltonian=basis@T@basis.T+np.diag(potential)
        info.update(metric_first_cm=float(metric_first_cm),metric_second_cm=float(metric_second_cm),
                    diagonal_metric=metric_info,quadratic_metric_positive_proved=True,kinetic_ordering='p G(q) p / 2')
    energies,vectors=eigh(hamiltonian,subset_by_index=(0,max_quanta),check_finite=False)
    excitation=energies-energies[0]
    shift=0.
    if fundamental_anchor_cm is not None:
        if not np.isfinite(fundamental_anchor_cm) or fundamental_anchor_cm<=0:raise ValueError('positive fundamental anchor required')
        shift=fundamental_anchor_cm-excitation[1]
        excitation+=shift*np.arange(max_quanta+1)
    if np.any(np.diff(excitation)<=0):raise ValueError('regularized/anchored mode table is not monotone')
    outer=abs(q)>.8*half_width
    return dict(excitation_levels_cm=excitation,variational_levels_cm=energies,
                outer_box_probability=np.sum(vectors[outer]**2,axis=0),
                linear_occupation_anchor_shift_cm=shift,grid_points=grid_points,
                half_width=half_width,potential_model=info,
                perturbative_absolute_ZPE_replaced=False)
