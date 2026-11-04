"""Native Hamiltonian fits to sampled spectra with analytic Voigt responses.

The measured channel signal, not extracted peak frequencies, is fitted.
Widths are Gaussian standard deviation and Lorentzian half width, in MHz.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from scipy.special import wofz


def voigt_response(offset_MHz, gaussian_sigma_MHz, lorentz_hwhm_MHz):
    """Unit-area profile and derivatives with respect to offset, sigma, gamma."""
    x=np.asarray(offset_MHz,float);s=float(gaussian_sigma_MHz);g=float(lorentz_hwhm_MHz)
    if not np.all(np.isfinite(x)) or not np.isfinite(s+g) or s<=0 or g<0:
        raise ValueError('finite offsets, positive Gaussian sigma and nonnegative Lorentz HWHM required')
    z=(x+1j*g)/(s*np.sqrt(2));w=wofz(z);dw=-2*z*w+2j/np.sqrt(np.pi)
    factor=1/(s*np.sqrt(2*np.pi))
    return (factor*w.real, factor*dw.real/(s*np.sqrt(2)),
            -factor*(w+z*dw).real/s, -factor*dw.imag/(s*np.sqrt(2)))


class _NativeLines:
    def __init__(self,surface,request):
        from .lam_global_fit import _backend
        from .assignment_spin import _SpinContext,_model,_parse
        request=deepcopy(request);request.setdefault('lam_states',1)
        if request.get('schema')!='matrix.lock.lam_spectral_fit.v1':raise ValueError('native LAM/spin fit request required')
        if surface.dipole_au is None and request.get('dipole_au') is not None:
            from dataclasses import replace
            surface=replace(surface,dipole_au=np.broadcast_to(request['dipole_au'],surface.grid_shape+(3,)).copy())
        self.request=request;self.lines=[]
        for row in request['observations']:
            if row.get('role','training')!='training':
                raise ValueError('profile components must be declared independently of held-out peak assignments')
            for component in row.get('components',[row]):
                self.lines.append(dict(lower=component['lower'],upper=component['upper'],
                    label=f'profile-line-{len(self.lines)}',frequency_MHz=0.,uncertainty_MHz=1.,
                    weight_mode='native' if request.get('profile_weight_mode','native')=='native' else 'fixed'))
        if not self.lines:raise ValueError('profile fit requires an explicit complete component list')
        # Duplicating components in overlapping blend groups would double the signal.
        def state_key(s):
            return (s.get('J'),s.get('two_F'),s.get('reference_index'),s.get('state_id'))
        keys=[(state_key(x['lower']),state_key(x['upper'])) for x in self.lines]
        if len(set(keys))!=len(keys):raise ValueError('profile component transitions must be unique')
        self.native=request.get('profile_weight_mode','native')=='native'
        if request.get('profile_weight_mode','native') not in ('native','fixed'):
            raise ValueError('profile_weight_mode must be native or fixed')
        weights=[c.get('weight',1.) for row in request['observations'] for c in row.get('components',[row])]
        self.weights=np.asarray(weights,float)
        if not np.all(np.isfinite(self.weights)) or np.any(self.weights<=0):raise ValueError('positive line weights required')
        if request.get('spin_model') is not None:
            model=_model(request['spin_model']);self.variables=_parse(surface,model,request)
            fs=sorted({c[side]['two_F'] for c in self.lines for side in ('lower','upper')})
            self.context=_SpinContext(surface,model,self.variables,[v['initial'] for v in self.variables],
                fs=fs,lam_states=request.get('lam_states',1),
                options={**request.get('options',{}),**({} if request.get('temperature_K') is None else {'temperature_K':request['temperature_K']})},
                operator_manifest=request.get('operator_manifest'))
            self.backend=None
        else:
            flat=deepcopy(request);flat.pop('profile_weight_mode',None);flat['observations']=self.lines
            # This backend supplies native line responses only. The complete
            # QM blocks, including profile coefficients, belong to the signal objective.
            for key in ('predicates','predicate_covariance','prior_covariance'):flat.pop(key,None)
            self.backend=_backend(surface,flat);self.variables=list(self.backend['variables']);self.context=None

    def evaluate(self,x):
        if self.context is not None:
            rows=self.context.evaluate(x,self.lines)
            frequencies=rows['values'];jac=rows['jacobian']
            strengths=np.array([r['strengths'][0] for r in rows['components']])
            ds=np.array([r['strength_jacobian'][0] for r in rows['components']])
        else:
            from .lam_measurements import native_strengths
            frequencies=self.backend['predict'](x)[0]
            jac=self.backend['jac'](x)[:len(self.lines)] if len(x) else np.zeros((len(self.lines),0))
            strengths,ds=(native_strengths(self.backend,x,np.arange(len(self.lines)),derivatives=bool(len(x)))
                          if self.native else (np.ones(len(self.lines)),np.zeros((len(self.lines),len(x)))))
        return np.asarray(frequencies),np.asarray(jac),strengths*self.weights,ds*self.weights[:,None]

    def promote_accepted(self,x):
        if self.context is not None:self.context.promote_accepted(x)
        else:self.backend['promote_accepted'](x)


class SpectralProfileContext:
    """Reusable analytic signal model; native line evaluation is cached upstream."""
    defaults={'amplitude':1.,'gaussian_sigma_MHz':.1,'lorentz_hwhm_MHz':0.,
              'baseline':0.,'slope_per_MHz':0.,'frequency_offset_MHz':0.}

    def __init__(self,line_model,frequency_MHz,profile,variables):
        self.line_model=line_model;self.x=np.asarray(frequency_MHz,float);self.profile=profile
        self.n=len(line_model.variables);self.variables=variables
        self.origin=float(profile.get('baseline_origin_MHz',np.mean(self.x)))
        response=profile.get('instrument',{})
        if set(response)-{'gaussian_sigma_MHz','offsets_MHz','weights'}:raise ValueError('unknown instrument response fields')
        self.instrument_sigma=float(response.get('gaussian_sigma_MHz',0.))
        self.offsets=np.asarray(response.get('offsets_MHz',[0.]),float)
        self.kernel=np.asarray(response.get('weights',[1.]),float)
        if (not np.isfinite(self.origin+self.instrument_sigma) or self.instrument_sigma<0
                or self.offsets.ndim!=1 or not len(self.offsets) or self.kernel.shape!=self.offsets.shape
                or not np.all(np.isfinite(self.offsets)) or not np.all(np.isfinite(self.kernel))
                or np.any(self.kernel<0) or self.kernel.sum()<=0):
            raise ValueError('instrument needs finite offsets and nonnegative normalized quadrature weights')
        self.kernel=self.kernel/self.kernel.sum()
        self.reference_strength=None

    def evaluate(self,values,observations=None):
        pars={**self.defaults,**self.profile.get('parameters',{})}
        for v,x in zip(self.variables[self.n:],values[self.n:],strict=True):pars[v['path'][8:]]=float(x)
        if set(pars)!=set(self.defaults) or not all(np.isfinite(v) for v in pars.values()):
            raise ValueError('unknown or nonfinite profile parameter')
        if pars['amplitude']<=0 or pars['gaussian_sigma_MHz']<=0 or pars['lorentz_hwhm_MHz']<0:
            raise ValueError('positive amplitude/Gaussian sigma and nonnegative Lorentz HWHM required')
        f,df,strengths,ds=self.line_model.evaluate(np.asarray(values[:self.n]))
        if (not np.all(np.isfinite(f)) or not np.all(np.isfinite(strengths)) or np.any(strengths<0)
                or strengths.sum()<=0):raise ValueError('finite positive total native line strength required')
        # Fixed initial normalization is absorbed by the fitted signal amplitude.
        if self.reference_strength is None:self.reference_strength=float(strengths.sum())
        strengths=strengths/self.reference_strength;ds=ds/self.reference_strength
        signal=np.zeros(len(self.x));jac=np.zeros((len(self.x),len(values)))
        nuisance={k:np.zeros(len(self.x)) for k in self.defaults}
        sigma=np.hypot(pars['gaussian_sigma_MHz'],self.instrument_sigma)
        for offset,k in zip(self.offsets,self.kernel,strict=True):
            for i,center in enumerate(f):
                p,dx,dsg,dg=voigt_response(self.x-center-pars['frequency_offset_MHz']-offset,sigma,pars['lorentz_hwhm_MHz'])
                p*=k;dx*=k;dsg*=k;dg*=k
                signal+=strengths[i]*p
                jac[:,:self.n]+=pars['amplitude']*(p[:,None]*ds[i]-strengths[i]*dx[:,None]*df[i])
                nuisance['frequency_offset_MHz']-=strengths[i]*dx
                nuisance['gaussian_sigma_MHz']+=strengths[i]*dsg*pars['gaussian_sigma_MHz']/sigma
                nuisance['lorentz_hwhm_MHz']+=strengths[i]*dg
        nuisance['amplitude']=signal.copy()
        nuisance['baseline'][:]=1.;nuisance['slope_per_MHz']=self.x-self.origin
        for key in ('frequency_offset_MHz','gaussian_sigma_MHz','lorentz_hwhm_MHz'):nuisance[key]*=pars['amplitude']
        signal=pars['amplitude']*signal+pars['baseline']+pars['slope_per_MHz']*(self.x-self.origin)
        for i,v in enumerate(self.variables[self.n:],self.n):jac[:,i]=nuisance[v['path'][8:]]
        return dict(values=signal,jacobian=jac,line_frequencies_MHz=f,line_strengths=strengths)

    def promote_accepted(self,values):self.line_model.promote_accepted(values[:self.n])


def fit_native_spectral_profile(surface,configuration):
    """Fit raw channel signals with native Hamiltonian and nuisance parameters.

    Channels with role other than training are only evaluated afterwards.
    Correlated channel covariance is marginalized to training channels.
    """
    from .assignment_statistics import fit_assignment_stage
    from .assignment_covariance import remap_assignment_covariance
    from .assignment_checkpoint import assignment_fingerprint
    config=deepcopy(configuration)
    if config.pop('schema',None)!='matrix.lock.spectral_profile_fit.v1':raise ValueError('unsupported profile fit schema')
    if set(config)-{'fit_request','profile','variables','max_nfev','variable_projection'}:raise ValueError('unknown profile fit fields')
    request=config['fit_request'];profile=config['profile'];channels=profile['channels']
    if set(profile)-{'channels','parameters','instrument','covariance','covariance_labels','baseline_origin_MHz'}:
        raise ValueError('unknown sampled profile fields')
    if request.get('options',{}).get('loss','linear')!='linear' or request.get('structured_discrepancy') is not None:
        raise ValueError('profile fitting currently requires an explicit Gaussian signal objective without discrepancy calibration')
    if request.get('covariance_MHz2') is not None:
        raise ValueError('peak-frequency covariance cannot be reused as channel-signal covariance')
    ids=[str(c['id']) for c in channels];x=np.array([c['frequency_MHz'] for c in channels],float)
    y=np.array([c['signal'] for c in channels],float);sigma=np.array([c['uncertainty'] for c in channels],float)
    if (not len(ids) or len(set(ids))!=len(ids) or not all(ids) or not np.all(np.isfinite(np.r_[x,y,sigma]))
            or np.any(sigma<=0) or np.any(np.diff(x)<=0)):
        raise ValueError('unique channel IDs, increasing finite frequencies, finite signals and positive errors required')
    train=np.array([i for i,c in enumerate(channels) if c.get('role','training')=='training'],int)
    if not len(train):raise ValueError('training channels required')
    native=_NativeLines(surface,request);variables=deepcopy(native.variables)
    for raw in config.get('variables',[]):
        path=raw['path'];name=path.removeprefix('profile.')
        if not path.startswith('profile.') or name not in SpectralProfileContext.defaults:raise ValueError('unknown profile variable')
        positive=name in ('amplitude','gaussian_sigma_MHz');nonnegative=name=='lorentz_hwhm_MHz'
        lo=max(float(raw.get('lower',-np.inf)),1e-12 if positive else 0. if nonnegative else -np.inf)
        v=dict(path=path,name=raw.get('name',path),initial=float(raw.get('initial',profile.get('parameters',{}).get(name,SpectralProfileContext.defaults[name]))),
            lower=lo,upper=float(raw.get('upper',np.inf)),scale=float(raw.get('scale',1.)),prior=raw.get('prior'))
        if not np.isfinite(v['initial']) or not np.isfinite(v['scale']) or v['scale']<=0 or not lo<v['upper'] or not lo<=v['initial']<=v['upper']:
            raise ValueError('invalid profile variable bounds or scale')
        variables.append(v)
    paths=[v['path'] for v in variables];names=[v['name'] for v in variables]
    if not paths or len(set(paths))!=len(paths) or len(set(names))!=len(names):raise ValueError('unique fit variable names and paths required')
    budget=config.get('max_nfev',30)
    if isinstance(budget,bool) or not isinstance(budget,int) or budget<1:raise ValueError('positive integer fit budget required')
    context=SpectralProfileContext(native,x,profile,variables)
    initial=np.array([v['initial'] for v in variables]);context.evaluate(initial)
    class Training:
        def evaluate(self,values,observations):
            data=context.evaluate(values);return dict(values=data['values'][train],jacobian=data['jacobian'][train])
        def promote_accepted(self,values):context.promote_accepted(values)
    statistical=deepcopy(request);statistical.pop('covariance_MHz2',None)
    covariance=profile.get('covariance')
    if covariance is not None:
        matrix,_=remap_assignment_covariance(covariance,profile.get('covariance_labels',ids),[ids[i] for i in train])
        statistical['covariance_MHz2']=matrix
    rows=[dict(frequency_MHz=float(y[i]),uncertainty_MHz=float(sigma[i])) for i in train]
    from .profile_projection import ProfileObjective,fit_projected_profile,normalize_profile_predicates
    projection=config.get('variable_projection',True)
    if not isinstance(projection,bool):raise ValueError('variable_projection must be a boolean')
    statistical=normalize_profile_predicates(variables,statistical)
    objective=ProfileObjective(context,train,y[train],sigma[train],variables,statistical)
    linear={'profile.amplitude','profile.baseline','profile.slope_per_MHz'}
    if projection and linear.intersection(paths):
        fit=fit_projected_profile(objective,variables,initial,statistical,max_nfev=budget)
    else:
        fit=fit_assignment_stage(Training(),rows,variables,initial,paths,statistical,max_nfev=budget)
        fit['variable_projection']=dict(enabled=False,reason='disabled' if not projection else 'no declared linear variables',
            nonlinear_dimension=len(variables),full_dimension=len(variables))
    fit['parameter_covariance']=objective.covariance(np.asarray(fit['values']),
        np.array([v['lower'] for v in variables]),np.array([v['upper'] for v in variables]))
    data=context.evaluate(fit['values']);prediction=data['values'];fit['rms_signal']=fit.pop('rms_residual_MHz')
    fit.pop('frequencies_MHz');fit.pop('jacobian')
    return dict(schema='matrix.lock.spectral_profile_result.v1',success=fit['success'],fit=fit,
        parameters=dict(zip(paths,fit['values'],strict=True)),
        channels=[dict(c,calculated=float(v),residual=float(v-c['signal'])) for c,v in zip(channels,prediction,strict=True)],
        line_frequencies_MHz=data['line_frequencies_MHz'].tolist(),line_strengths=data['line_strengths'].tolist(),
        configuration_sha256=assignment_fingerprint(surface,configuration),
        scope='native component list and isolated-line Voigt superposition; no line mixing, saturation or collision-induced coherence; held-out channels excluded from fitting')


__all__=['voigt_response','SpectralProfileContext','fit_native_spectral_profile']
