"""Finite active-state operator traces combined with an analytic Gaussian bath.

The active coordinate operators act in the same orthonormal coefficient
space on both surfaces. Signed mixed traces retain FC/HT interference.
No active/bath Hamiltonian coupling is silently inferred by this contraction.
"""
import numpy as np
from scipy.fft import fftshift,hfft,fftfreq
from .vibronic_harmonic import PHASE_PER_CM_FS


def sine_position_matrix(count,bounds):
    """Exact position matrix in an orthonormal Dirichlet sine basis."""
    lo,hi=map(float,bounds)
    if count<1 or not hi>lo:raise ValueError('Positive sine dimension and ordered bounds required')
    n=np.arange(1,count+1);minus=n[:,None]-n[None,:];plus=n[:,None]+n[None,:]
    denominator=np.where(minus==0,1,minus)**2
    matrix=(hi-lo)/np.pi**2*(((-1.)**minus-1)/denominator-((-1.)**plus-1)/plus**2)
    np.fill_diagonal(matrix,(lo+hi)/2)
    return matrix


def project_sine_linear_operators(initial,final,basis_counts,bounds,constants,gradients):
    """Project arbitrary real linear operators between two product sine spaces."""
    vi=np.asarray(initial);vf=np.asarray(final);shape=tuple(basis_counts);c=np.asarray(constants);g=np.asarray(gradients)
    if vi.shape[0]!=np.prod(shape) or vf.shape[0]!=vi.shape[0] or g.shape!=(len(c),len(shape)):raise ValueError('Matching sine coefficients and linear operators required')
    overlap=vf.T@vi;amplitudes=c[:,None,None]*overlap
    for j,count in enumerate(shape):
        matrix=sine_position_matrix(count,bounds[j]);tensor=vi.reshape(shape+(vi.shape[1],));action=np.moveaxis(np.tensordot(matrix,tensor,axes=(1,j)),0,j).reshape(vi.shape);projected=vf.T@action;amplitudes+=g[:,j,None,None]*projected
    return overlap,amplitudes


def active_dipole_traces(initial_energies,final_energies,overlap,dipole_amplitudes,
                         populations,times_fs,*,energy_bin_cm=1.):
    """Return scalar, dipole-norm and three mixed active spectral traces.

    Real bound-state amplitudes permit a five-form bath contraction: the bath
    mixed forms already include both Hermitian orders of the insertion.
    """
    ei=np.asarray(initial_energies,float);ef=np.asarray(final_energies,float);o=np.asarray(overlap);mu=np.asarray(dipole_amplitudes);p=np.asarray(populations,float);t=np.asarray(times_fs,float)
    if (o.shape!=(len(ef),len(ei)) or mu.shape!=(3,)+o.shape or p.shape!=ei.shape
            or np.iscomplexobj(o) or np.iscomplexobj(mu) or np.any(p<0)
            or energy_bin_cm<=0 or not all(np.all(np.isfinite(x)) for x in (ei,ef,o,mu,p,t))):
        raise ValueError('Compatible finite real active amplitudes and populations required')
    energy=ei[None,:]-ef[:,None];low=np.floor(energy.min()/energy_bin_cm)*energy_bin_cm
    high=np.ceil(energy.max()/energy_bin_cm)*energy_bin_cm+energy_bin_cm
    bins=np.arange(low,high+energy_bin_cm/2,energy_bin_cm);center=(bins[:-1]+bins[1:])/2
    weights=[o*o,np.sum(mu*mu,axis=0),*(mu[j]*o for j in range(3))]
    hist=np.array([np.histogram(energy,bins,weights=w*p[None,:])[0] for w in weights]);live=np.flatnonzero(np.any(hist!=0,axis=0));traces=np.zeros((5,len(t)),complex)
    for ids in np.array_split(live,max(1,(len(live)+255)//256)):
        traces+=hist[:,ids]@np.exp(-1j*PHASE_PER_CM_FS*center[ids,None]*t)
    return traces,dict(scalar_strength=float(hist[0].sum()),dipole_strength=float(hist[1].sum()),energy_bin_cm=energy_bin_cm)


def product_dipole_correlation(active,bath):
    """Contract active [I,|mu|^2,mu_j] with bath [I,|B|^2,IB_j+B_jI]."""
    a=np.asarray(active);b=np.asarray(bath)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=5:raise ValueError('Five matching active/bath trace forms required')
    return dict(FC=a[1]*b[0],HT_bath=a[0]*b[1],interference=np.sum(a[2:]*b[2:],axis=0))


def broadened_trace(correlation,times_fs,origin_cm,*,sigma_cm=100.,fft_padding_factor=1):
    t=np.asarray(times_fs,float);dt=t[1]-t[0]
    if len(t)<3 or t[0]!=0 or dt<=0 or not np.allclose(np.diff(t),dt):raise ValueError('Uniform nonnegative time grid required')
    spacing=PHASE_PER_CM_FS*dt;signal=np.asarray(correlation)*np.exp(-.5*(sigma_cm*PHASE_PER_CM_FS*t)**2)
    length=2*(len(t)-1)*fft_padding_factor
    density=fftshift(hfft(signal.conj(),n=length))*spacing/(2*np.pi)
    axis=fftshift(2*np.pi*fftfreq(len(density),d=spacing))+origin_cm
    return axis,density
