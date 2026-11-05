"""Shared spectral-window goals for adaptive TI and analytic TD references.

Window recovery is measured against an unrenormalized numerical TD reference;
it is not a rigorous replacement for the global positive-norm certificate.
"""
import hashlib
import numpy as np
from scipy.special import ndtr
from .vibronic_harmonic import harmonic_thermal_factors


class SpectralSelectionTarget:
    """Gaussian/Lorentz window areas, automatic ranking and profile diagnostics."""

    def __init__(self, windows_cm, axis_cm, densities, bank_density, *,
                 gaussian_sigma_cm, lorentz_hwhm_cm=0., profile_points=65,
                 profile_peak_tolerance=.02,profile_L1_tolerance=.01,
                 profile_absolute_tolerance=0.,window_rankings=True,assignment_limit=50,rate_at_cm=None):
        windows=np.asarray(windows_cm,float);x=np.asarray(axis_cm,float)
        y=np.asarray(densities,float);bank=np.asarray(bank_density,float)
        if (windows.ndim!=2 or windows.shape[1]!=2 or not len(windows)
                or not np.all(np.isfinite(windows)) or np.any(windows[:,0]>=windows[:,1])):
            raise ValueError('nonempty finite spectral windows [lower, upper] required')
        if (x.ndim!=1 or len(x)<3 or np.any(np.diff(x)<=0) or not np.all(np.isfinite(x))
                or y.ndim!=2 or y.shape[1]!=len(x) or bank.shape!=x.shape
                or not np.all(np.isfinite(y)) or not np.all(np.isfinite(bank))):
            raise ValueError('finite ordered TD grid and compatible operator/bank densities required')
        if (not np.isfinite(gaussian_sigma_cm+lorentz_hwhm_cm)
                or min(gaussian_sigma_cm,lorentz_hwhm_cm)<0
                or gaussian_sigma_cm+lorentz_hwhm_cm<=0
                or gaussian_sigma_cm*lorentz_hwhm_cm!=0):
            raise ValueError('targeted selection requires a pure Gaussian or pure Lorentz line shape')
        if isinstance(profile_points,bool) or not isinstance(profile_points,int) or profile_points<3:
            raise ValueError('at least three comparison points per window required')
        windows=windows[np.argsort(windows[:,0],kind='stable')]
        if np.any(windows[1:,0]<windows[:-1,1]):
            raise ValueError('target spectral windows must not overlap')
        if np.any(windows[:,0]<=x[0]) or np.any(windows[:,1]>=x[-1]):
            raise ValueError('target windows must lie inside the TD spectral grid')
        if np.min(windows[:,1]-windows[:,0])<8*np.max(np.diff(x)):
            raise ValueError('TD grid is too coarse for the requested window; increase time_points')
        peak=float(np.max(bank,initial=0.))
        if peak<=0 or np.min(bank)<-1e-6*peak:
            raise ValueError('positive TD bank density required; refine the TD reference')
        self.windows=windows;self.axis=x;self.reference=y;self.bank=bank
        self.sigma=float(gaussian_sigma_cm);self.gamma=float(lorentz_hwhm_cm)
        self.grid=np.array([np.linspace(lo,hi,profile_points) for lo,hi in windows])
        spacing=x[1]-x[0]
        if not np.allclose(np.diff(x),spacing,rtol=1e-8,atol=1e-9):
            raise ValueError('uniform TD Fourier grid required')
        # Integrate/evaluate the TD trigonometric polynomial, rather than
        # interpolating narrow bands linearly between FFT nodes.
        self._fourier=np.fft.fft(np.vstack([y,bank]),axis=1)/len(x)
        self._omega=2*np.pi*np.fft.fftfreq(len(x),d=spacing)
        self.reference_profiles=np.array([self._reference_at(grid)[:-1] for grid in self.grid])
        shape=(len(windows),len(y))
        def tolerance(value,name):
            try:result=np.broadcast_to(np.asarray(value,float),shape).copy()
            except ValueError as error:raise ValueError(name+' must broadcast over windows and observables') from error
            if np.any(~np.isfinite(result)) or np.any(result<0):raise ValueError(name+' must be finite and nonnegative')
            return result
        self.peak_tolerance=tolerance(profile_peak_tolerance,'profile peak tolerance')
        self.L1_tolerance=tolerance(profile_L1_tolerance,'profile L1 tolerance')
        self.absolute_tolerance=tolerance(profile_absolute_tolerance,'profile absolute tolerance')
        if not isinstance(window_rankings,bool):raise ValueError('window_rankings must be boolean')
        if isinstance(assignment_limit,bool) or not isinstance(assignment_limit,int) or assignment_limit<1:
            raise ValueError('positive assignment limit required')
        self.window_rankings=window_rankings;self.assignment_limit=assignment_limit
        self.observable_norms=np.ones(len(y));self.line_count=0
        self.computed_profiles=np.zeros_like(self.reference_profiles)
        self.reference_bank_profiles=np.array([self._reference_at(grid)[-1] for grid in self.grid])
        self.reference_areas=[];self.reference_bank=[]
        self._window_means=[]
        self._guide_energy=0.;self._guide_mass=0.
        for lo,hi in windows:
            xx=np.r_[lo,x[(x>lo)&(x<hi)],hi]
            yy=np.array([np.interp(xx,x,row) for row in y]);bb=np.interp(xx,x,bank)
            factors=np.empty(len(x),complex);factors[0]=hi-lo
            omega=self._omega[1:]
            factors[1:]=(np.exp(1j*omega*(hi-x[0]))-np.exp(1j*omega*(lo-x[0])))/(1j*omega)
            integrals=(self._fourier@factors).real
            self.reference_areas.append(integrals[:-1])
            self.reference_bank.append(integrals[-1])
            mass=np.trapezoid(np.maximum(bb,0.),xx)
            self._window_means.append(np.trapezoid(xx*np.maximum(bb,0.),xx)/mass if mass else (lo+hi)/2)
            self._guide_energy+=np.trapezoid(xx*np.maximum(bb,0.),xx)
            self._guide_mass+=np.trapezoid(np.maximum(bb,0.),xx)
        self.reference_areas=np.asarray(self.reference_areas);self.reference_bank=np.asarray(self.reference_bank)
        if np.any(self.reference_bank<=0):raise ValueError('each target window needs positive reference strength')
        self.computed_bank=np.zeros(len(windows));self.computed_areas=np.zeros_like(self.reference_areas)
        self.rate_at_cm=None if rate_at_cm is None else float(rate_at_cm)
        if self.rate_at_cm is not None:
            if (not np.isfinite(self.rate_at_cm) or len(windows)!=1
                    or not windows[0,0]<=self.rate_at_cm<=windows[0,1]
                    or np.any(y< -1e-6*peak)):
                raise ValueError('rate targeting requires one window containing the point and positive strengths')
            self.rate_reference=self._reference_at(np.array([self.rate_at_cm]))[:-1,0]
            if np.any(self.rate_reference<=0):raise ValueError('positive TD density required at the rate target')
            self.rate_computed=np.zeros(len(y))
        digest=hashlib.sha256()
        for value in (windows,x,y,bank,np.array([self.sigma,self.gamma,profile_points])):
            digest.update(np.ascontiguousarray(value,dtype='<f8').tobytes())
        if self.rate_at_cm is not None:digest.update(np.asarray([self.rate_at_cm],dtype='<f8').tobytes())
        self.identity=digest.hexdigest()

    def _reference_at(self,grid):
        result=np.empty((len(self._fourier),len(grid)))
        chunk=max(1,min(64,8*1024**2//(16*len(self._omega))))
        for start in range(0,len(grid),chunk):
            stop=start+chunk
            result[:,start:stop]=(self._fourier@np.exp(1j*self._omega[:,None]*(grid[None,start:stop]-self.axis[0]))).real
        return result

    def _cdf(self,delta):
        return ndtr(delta/self.sigma) if self.sigma else .5+np.arctan(delta/self.gamma)/np.pi

    def _kernel(self,delta):
        return (np.exp(-.5*(delta/self.sigma)**2)/(np.sqrt(2*np.pi)*self.sigma)
            if self.sigma else self.gamma/(np.pi*(delta*delta+self.gamma*self.gamma)))

    def add(self, energies, norms, strengths):
        energies=np.asarray(energies);strengths=np.asarray(strengths)
        chunk=max(1,min(8192,8*1024**2//(8*self.grid.size)))
        for start in range(0,len(energies),chunk):
            stop=start+chunk;x=energies[start:stop]
            weights=self._cdf(self.windows[:,1,None]-x)-self._cdf(self.windows[:,0,None]-x)
            self.computed_bank+=weights@norms[start:stop]
            self.computed_areas+=weights@strengths[:,start:stop].T
            delta=x[:,None]-self.grid.ravel()
            profile=strengths[:,start:stop]@self._kernel(delta)
            self.computed_profiles+=profile.reshape(strengths.shape[0],*self.grid.shape).transpose(1,0,2)
            if self.rate_at_cm is not None:
                if np.any(strengths[:,start:stop]<0):raise ValueError('rate targeting requires positive line strengths')
                self.rate_computed+=strengths[:,start:stop]@self._kernel(x-self.rate_at_cm)
        self.line_count+=len(energies)

    def profile_status(self):
        difference=abs(self.computed_profiles-self.reference_profiles)
        peak=np.max(abs(self.reference_profiles),axis=2)
        l1=np.trapezoid(abs(self.reference_profiles),self.grid[:,None,:],axis=2)
        floor=256*np.finfo(float).eps*np.max(abs(self.reference_bank_profiles),axis=1)[:,None]*self.observable_norms
        absolute=self.absolute_tolerance+floor
        peak_error=np.max(difference,axis=2)
        l1_error=np.trapezoid(difference,self.grid[:,None,:],axis=2)
        ready=(peak_error<=self.peak_tolerance*peak+absolute)&(l1_error<=self.L1_tolerance*l1
            +absolute*np.diff(self.windows,axis=1))
        return ready,peak_error,l1_error,peak,l1

    def window_ready(self,fraction):
        if self.rate_at_cm is not None:
            return np.array([bool(np.all(self.rate_computed>=fraction*self.rate_reference)
                and np.all(self.rate_computed<=(2-fraction)*self.rate_reference))])
        return (self.computed_bank>=fraction*self.reference_bank)&np.all(self.profile_status()[0],axis=1)

    def converged(self,fraction):
        return bool(np.all(self.window_ready(fraction)))

    def automatic_bias(self, model, gaussian, temperature_K, direction,window_index=None):
        # Linear-response exponential tilt, bounded by three inverse window
        # widths. No TI pilot fits or molecular constants enter this choice.
        live=self.bank>1e-12*max(self.bank);x=self.axis[live];p=self.bank[live]
        mean=np.average(x,weights=p);variance=np.average((x-mean)**2,weights=p)
        target=self._guide_energy/self._guide_mass if window_index is None else self._window_means[window_index]
        cap=3./(np.min(np.diff(self.windows,axis=1)) if window_index is None else np.diff(self.windows[window_index])[0])
        bias=float(np.clip(direction*(mean-target)/variance,-cap,cap)) if variance else 0.
        q=harmonic_thermal_factors(model.initial_frequencies_cm,temperature_K)
        frequencies=np.r_[-model.initial_frequencies_cm,model.final_frequencies_cm]
        base=np.r_[np.sqrt(q),np.ones(len(q))]
        for _ in range(64):
            exponent=-.5*bias*frequencies
            if np.max(abs(exponent))<=300:
                scale=base*np.exp(exponent);matrix=scale[:,None]*gaussian['matrix']*scale[None,:]
                if np.max(abs(np.linalg.eigvalsh(matrix)))<1-1e-10:
                    if self.rate_at_cm is None:return bias
                    return self._rate_bias(model,gaussian,base,frequencies,direction,bias,cap)
            bias*=.5
        raise ValueError('could not construct a normalizable targeted ranking')

    def _rate_bias(self,model,gaussian,base,frequencies,direction,bias,cap):
        """Damped moment matching of the Gaussian energy-shell proposal.

        Only ranking changes. The common recurrence still evaluates the exact
        physical amplitudes for every operator component and selected state.
        """
        eye=np.eye(len(base));target=direction*self.rate_at_cm-model.origin_cm
        def moments(value):
            exponent=-.5*value*frequencies
            if np.max(abs(exponent))>300:return None
            scale=base*np.exp(exponent);a=scale[:,None]*gaussian['matrix']*scale[None,:]
            if np.max(abs(np.linalg.eigvalsh(a)))>=1-1e-10:return None
            vx=.5*np.linalg.solve(eye-a,eye+a);vp=.5*np.linalg.solve(eye+a,eye-a)
            mean=np.sqrt(2)*np.linalg.solve(eye-a,gaussian['linear']*scale)
            number=(np.diag(vx)+np.diag(vp)+mean*mean-1)/2
            covariance=.5*(vx*vx+vp*vp)+mean[:,None]*vx*mean[None,:]
            covariance[np.diag_indices(len(base))]-=.25
            return float(frequencies@number),max(0.,float(frequencies@covariance@frequencies))
        current=moments(bias)
        for _ in range(12):
            error=current[0]-target
            if abs(error)<1. or current[1]<=0:break
            step=error/current[1]
            for _ in range(16):
                trial=float(np.clip(bias+step,-cap,cap));candidate=moments(trial)
                if candidate is not None and abs(candidate[0]-target)<abs(error):
                    bias=trial;current=candidate;break
                step*=.5
            else:break
        self.rate_ranking_mean_cm=float(direction*(model.origin_cm+current[0]))
        return bias

    def report(self, energies, strengths, observable_norms):
        if self.line_count!=len(energies):raise ValueError('profile accumulator and assigned lines differ')
        self.observable_norms=np.asarray(observable_norms)
        profiles=self.computed_profiles
        ready,peak_error,l1_error,peak,l1=self.profile_status()
        missing=np.maximum(0.,self.reference_bank-self.computed_bank)
        result=dict(windows_cm=self.windows,gaussian_sigma_cm=self.sigma,lorentz_hwhm_cm=self.gamma,
            reference_bank_areas=self.reference_bank,computed_bank_areas=self.computed_bank,
            bank_recovery_fraction=self.computed_bank/self.reference_bank,
            reference_observable_areas=self.reference_areas,computed_observable_areas=self.computed_areas,
            reference_based_observable_omission_estimates=missing[:,None]*np.asarray(observable_norms),
            profile_wavenumbers_cm=self.grid,reference_profiles=self.reference_profiles,ti_profiles=profiles,
            profile_reference_resolved=peak>0,
            profile_peak_tolerance=self.peak_tolerance,profile_L1_tolerance=self.L1_tolerance,
            profile_absolute_tolerance=self.absolute_tolerance,profile_criteria_met=ready,
            profile_roundoff_floor=256*np.finfo(float).eps*np.max(abs(self.reference_bank_profiles),axis=1)[:,None]*self.observable_norms,
            profile_max_absolute_error=peak_error,
            profile_relative_peak_error=np.divide(peak_error,peak,out=np.zeros_like(peak),where=peak>0),
            profile_relative_L1_error=np.divide(l1_error,l1,
                out=np.zeros_like(l1),where=l1>0),
            reference='same-model analytic TD; exact periodic Fourier interpolation/integration; no normalization or fitted shift',
            TD_time_step_and_aliasing_require_separate_qualification=True,
            reference_sha256=self.identity,rigorous_global_certificate=False,
            physical_strengths_reweighted=False,signed_strengths_clipped=False)
        if self.rate_at_cm is not None:
            result['rate_target']=dict(wavenumber_cm=self.rate_at_cm,
                reference_density=self.rate_reference,computed_density=self.rate_computed,
                recovery_fraction=self.rate_computed/self.rate_reference,
                ranking_gaussian_mean_cm=getattr(self,'rate_ranking_mean_cm',None),
                rigorous_bound=False,scope='Pointwise positive density; not full-window or global completeness')
        return result


__all__=['SpectralSelectionTarget']
