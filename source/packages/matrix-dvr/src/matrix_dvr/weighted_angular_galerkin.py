"""Positive finite-interval Gaussian quadrature and pruned angular products.

Hydrogen-like angles use orthogonal polynomials times a Gaussian and an exact
Dirichlet wall factor. The Gaussian is a basis envelope, not a shortened
integration domain. The remaining periodic angle uses a Fourier basis.
"""
from itertools import product
import numpy as np


def finite_gaussian_axis(half_width, envelope_width, polynomial_count, quadrature_count, *, moment_order=256):
    L=float(half_width);sigma=float(envelope_width);n=int(polynomial_count);nq=int(quadrature_count)
    if (not 0<sigma<L or n<1 or nq<n or moment_order<4*nq
            or not np.isfinite(L) or not np.isfinite(sigma)):
        raise ValueError('Resolved positive finite Gaussian quadrature required')
    t,w=np.polynomial.legendre.leggauss(moment_order);x=L*t
    weight=L*w*(1-(x/L)**2)**2*np.exp(-(x/sigma)**2)
    mass=float(weight.sum());previous=np.zeros_like(x);current=np.ones_like(x)/np.sqrt(mass)
    alpha=[];beta=[0.]
    for k in range(nq):
        a=float(np.dot(weight*x,current**2));alpha.append(a)
        residual=(x-a)*current-beta[k]*previous
        b=float(np.sqrt(np.dot(weight,residual**2)));beta.append(b)
        if not b>1e-14:raise ValueError('Degenerate finite Gaussian moment recurrence')
        previous,current=current,residual/b
    J=np.diag(alpha)+np.diag(beta[1:nq],1)+np.diag(beta[1:nq],-1)
    q,U=np.linalg.eigh(J);wg=mass*U[0]**2
    P=np.zeros((nq,n));D=np.zeros_like(P);P[:,0]=1/np.sqrt(mass)
    for k in range(n-1):
        P[:,k+1]=((q-alpha[k])*P[:,k]-(beta[k]*P[:,k-1] if k else 0))/beta[k+1]
        D[:,k+1]=(P[:,k]+(q-alpha[k])*D[:,k]-(beta[k]*D[:,k-1] if k else 0))/beta[k+1]
    f=(1-(q/L)**2)*np.exp(-.5*(q/sigma)**2)
    logarithmic=-2*q/(L*L-q*q)-q/(sigma*sigma)
    values=P*np.sqrt(wg)[:,None]
    momenta=-1j*(D+logarithmic[:,None]*P)*np.sqrt(wg)[:,None]
    return dict(points=q,physical_weights=wg/f**2,values=values,momenta=momenta,
        measure_weights=wg,polynomials=P,recurrence_alpha=np.array(alpha),recurrence_beta=np.array(beta),
        physical_bounds=[-L,L],basis_envelope_width=sigma,
        orthogonality_error=float(np.max(abs(values.T@values-np.eye(n)))))


def pruned_gaussian_fourier_quadrature(hydrogen_half_widths, envelope_widths, total_degree,
                                      hydrogen_quadratures, fourier_order, periodic_quadrature, *,
                                      fourier_representation='complex'):
    limits=np.asarray(hydrogen_half_widths,float);widths=np.asarray(envelope_widths,float)
    nh=len(limits);degree=int(total_degree);kw=int(fourier_order);nw=int(periodic_quadrature)
    if widths.shape!=limits.shape or len(hydrogen_quadratures)!=nh or degree<0 or kw<0 or nw<=2*kw:
        raise ValueError('Compatible pruned polynomial/Fourier basis required')
    axes=[finite_gaussian_axis(L,s,degree+1,nq) for L,s,nq in zip(limits,widths,hydrogen_quadratures)]
    theta=(np.arange(nw)-nw//2)*2*np.pi/nw
    k=np.arange(-kw,kw+1)
    if fourier_representation=='complex':
        periodic=np.exp(1j*theta[:,None]*k[None,:])/np.sqrt(nw)
        periodic_momentum=periodic*k[None,:]
    elif fourier_representation=='real':
        periodic=np.empty((nw,len(k)));periodic_momentum=np.zeros((nw,len(k)),complex)
        for j,order in enumerate(k):
            if order==0:periodic[:,j]=1/np.sqrt(nw)
            elif order<0:
                periodic[:,j]=np.sqrt(2/nw)*np.sin(abs(order)*theta)
                periodic_momentum[:,j]=-1j*np.sqrt(2/nw)*abs(order)*np.cos(abs(order)*theta)
            else:
                periodic[:,j]=np.sqrt(2/nw)*np.cos(order*theta)
                periodic_momentum[:,j]=1j*np.sqrt(2/nw)*order*np.sin(order*theta)
    else:raise ValueError('Fourier representation must be complex or real')
    polynomials=[p for p in product(range(degree+1),repeat=nh) if sum(p)<=degree]
    labels=np.array([(*p,int(j)) for p in polynomials for j in k],int)
    nodes=[a['points'] for a in axes]+[theta]
    indices=np.stack(np.meshgrid(*[np.arange(len(x)) for x in nodes],indexing='ij'),axis=-1).reshape(-1,nh+1)
    q=np.stack([nodes[i][indices[:,i]] for i in range(nh+1)],axis=-1)
    factors=[axes[i]['values'][indices[:,i,None],labels[None,:,i]] for i in range(nh)]
    factors.append(periodic[indices[:,-1,None],labels[None,:,-1]+kw])
    u=np.prod(np.stack(factors),axis=0);momenta=[]
    for i in range(nh+1):
        f=factors.copy()
        f[i]=(axes[i]['momenta'][indices[:,i,None],labels[None,:,i]] if i<nh
              else periodic_momentum[indices[:,-1,None],labels[None,:,-1]+kw])
        momenta.append(np.prod(np.stack(f),axis=0))
    physical_weights=np.prod(np.stack([axes[i]['physical_weights'][indices[:,i]] for i in range(nh)]),axis=0)*2*np.pi/nw
    # The full tensor quadrature factorizes exactly, including after pruning.
    # Form its Gram from the small axis Grams instead of a tall dense product.
    overlap=np.ones((len(labels),len(labels)),dtype=periodic.dtype)
    for axis in range(nh):
        gram=axes[axis]['values'].T@axes[axis]['values']
        overlap*=gram[labels[:,axis,None],labels[None,:,axis]]
    gram=periodic.conj().T@periodic
    overlap*=gram[labels[:,-1,None]+kw,labels[None,:,-1]+kw]
    error=float(np.max(abs(overlap-np.eye(len(labels)))))
    if error>1e-9:raise ValueError('Quadrature does not resolve the pruned angular basis')
    return dict(points=q,weights=physical_weights,values=u,momenta=np.stack(momenta,axis=1),
        occupations=labels,orthogonality_error=error,axes=axes,
        physical_bounds=np.array([[-L,L] for L in limits]+[[-np.pi,np.pi]]),
        basis_envelope_widths=widths,total_degree=degree,fourier_order=kw,
        fourier_representation=fourier_representation)
