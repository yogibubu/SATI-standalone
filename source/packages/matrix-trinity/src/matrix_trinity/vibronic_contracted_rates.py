"""Conditional vibronic sources for existing contracted native mode products."""
from collections import OrderedDict
import hashlib
import numpy as np

from .vibronic_harmonic import orient_harmonic_model
from .vibronic_spectator_rates import _SpectatorOverlapKernel


class ContractedSpectatorObservable:
    """Contract a core/group native spectrum without a full HO product matrix.

    The supplied initial state is embedded in the oriented initial modal frame.
    Spectators remain harmonic in the overlap; explicit independent eigenenergy
    corrections can reuse the conditional doorways without changing the source.
    """
    def __init__(self,model,initial,core_basis,core_modes,block_basis,block_modes,
                 block_vectors,product,prepared_spectrum,coupling_hartree,*,
                 kind='emission',gaussian_sigma_cm=21.23,matrix_zero_tolerance=0.,
                 maximum_harmonic_pairs=50_000_000,maximum_grid_entries=2_000_000_000,
                 maximum_workspace_bytes=8*1024**3,maximum_cached_states=4096,workers=1):
        from .vibronic_contracted_basis import ContractedModeProduct
        from .vibronic_spectral_cache import PreparedGaussianSpectrum
        if not isinstance(product,ContractedModeProduct) or not isinstance(prepared_spectrum,PreparedGaussianSpectrum):
            raise TypeError('existing contracted product and prepared spectrum required')
        oriented,_=orient_harmonic_model(model,kind)
        n=len(oriented.initial_frequencies_cm)
        cb,bb=np.asarray(core_basis),np.asarray(block_basis)
        cm,bm=np.asarray(core_modes),np.asarray(block_modes)
        vectors=np.asarray(block_vectors);coupling=complex(coupling_hartree)
        if coupling.imag==0:coupling=coupling.real
        if (cm.ndim!=1 or bm.ndim!=1 or cm.dtype.kind not in 'iu' or bm.dtype.kind not in 'iu'
                or not len(cm) or not len(bm) or cb.ndim!=2 or bb.ndim!=2
                or cb.dtype.kind not in 'iu' or bb.dtype.kind not in 'iu'
                or cb.shape!=(product.core_vectors.shape[0],len(cm))
                or bb.shape[1]!=len(bm) or np.any(cb<0) or np.any(bb<0)
                or len(np.unique(np.r_[cm,bm]))!=len(cm)+len(bm)
                or np.any(np.r_[cm,bm]<0) or np.any(np.r_[cm,bm]>=n)
                or len(cm)+len(bm)>=n or vectors.shape!=(len(bb),len(product.counts))
                or not np.all(np.isfinite(vectors))
                or not np.allclose(vectors.conj().T@vectors,np.eye(len(product.counts)),atol=1e-10,rtol=1e-10)
                or prepared_spectrum.eigenvectors.shape[0]!=product.size
                or not np.array_equal(initial.frequencies_cm,oriented.initial_frequencies_cm)
                or not np.isfinite(coupling) or not np.isfinite(gaussian_sigma_cm) or gaussian_sigma_cm<=0
                or isinstance(workers,bool) or not isinstance(workers,int) or workers<1
                or isinstance(maximum_cached_states,bool) or not isinstance(maximum_cached_states,int)
                or maximum_cached_states<1):
            raise ValueError('matching finite modal product, disjoint active indices and positive budgets required')
        full=np.zeros((len(cb)*len(bb),n),int)
        full[:,cm]=np.tile(cb,(len(bb),1));full[:,bm]=np.repeat(bb,len(cb),axis=0)
        kernel=_SpectatorOverlapKernel(oriented,initial.basis,full,
            matrix_zero_tolerance=matrix_zero_tolerance,maximum_grid_entries=maximum_grid_entries,
            maximum_harmonic_pairs=maximum_harmonic_pairs,cache_components=True)
        zero=np.flatnonzero(np.all(initial.basis==0,axis=1))
        if len(zero)!=1:raise ValueError('one initial harmonic vacuum required')
        coefficients=initial.eigenvectors[:,0]
        if not np.any(coefficients.imag):coefficients=coefficients.real
        self.contraction=kernel.prepare_contraction(coefficients,int(zero[0]),maximum_workspace_bytes=maximum_workspace_bytes)
        self.kernel,self.product,self.prepared_spectrum=kernel,product,prepared_spectrum
        self.initial,self.oriented=initial,oriented
        self.spectator_modes=np.setdiff1d(np.arange(n),np.r_[cm,bm])
        self.frequencies_cm=oriented.final_frequencies_cm[self.spectator_modes]
        self.block_vectors=vectors.copy();self.core_rows=len(cb);self.block_rows=len(bb)
        self.coupling_hartree=coupling;self.sigma=float(gaussian_sigma_cm);self.workers=workers
        self.maximum_batch_size=8;self.uniform_density_tolerance=1e-18
        self.maximum_cached_states=maximum_cached_states;self._cache=OrderedDict()
        self._last_occupations=None;self._last_doorway=None
        self.initial_energy_cm=float(initial.energies_cm[0]-oriented.electronic_gap_cm)
        digest=hashlib.sha256(b'matrix.contracted_spectator.v1')
        for value in (kernel.matrix,kernel.linear,np.array([kernel.ground]),initial.basis,
                initial.energies_cm,initial.eigenvectors,cb,cm,bb,bm,vectors,
                product.core_energies_cm,product.core_vectors,product.extra_energies_cm,product.counts,
                self.frequencies_cm,np.array([self.initial_energy_cm,self.sigma]),np.array([coupling])):
            value=np.ascontiguousarray(value);digest.update(str((value.dtype.str,value.shape)).encode());digest.update(value.tobytes())
        digest.update(prepared_spectrum.hamiltonian_sha256.encode())
        self.persistent_model_key=digest.hexdigest()
        self.active_hamiltonian_sha256=prepared_spectrum.hamiltonian_sha256
        self.diagnostics=dict(model='coupled native contracted product with harmonic Duschinsky spectators',
            active_hamiltonian_sha256=self.active_hamiltonian_sha256,model_key=self.persistent_model_key,
            source_wavefunctions_changed=False,retained_group_levels=len(product.counts),
            contracted_states=product.size,independent_anharmonic_spectators_included=False,
            conditional_packet_cache_hits=0,conditional_packet_evaluations=0)

    def conditional_doorways(self,occupations):
        rows=np.asarray(occupations)
        if (rows.ndim!=2 or rows.shape[1]!=len(self.spectator_modes)
                or rows.dtype.kind not in 'iu' or np.any(rows<0)):
            raise ValueError('nonnegative integer spectator occupation rows required')
        if self._last_occupations is not None and np.array_equal(rows,self._last_occupations):
            self.diagnostics['conditional_packet_cache_hits']+=1
            return dict(self._last_doorway)
        sources=[]
        for row in rows:
            raw,_=self.contraction(self.spectator_modes,row)
            modal=self.block_vectors.conj().T@raw.reshape(self.block_rows,self.core_rows)
            sources.append(self.coupling_hartree*self.product.project(modal.reshape(-1,1))[:,0])
        packet=np.column_stack(sources) if sources else np.empty((self.product.size,0),complex)
        energy=(rows+.5)@self.frequencies_cm
        packet.setflags(write=False);energy.setflags(write=False)
        result=dict(source_hartree=packet,initial_energy_cm=self.initial_energy_cm,spectator_energies_cm=energy)
        self._last_occupations=rows.copy();self._last_doorway=result
        self.diagnostics['conditional_packet_evaluations']+=len(rows)
        return dict(result)

    def anharmonic(self,occupations):
        rows=np.asarray(occupations)
        if (rows.ndim!=2 or rows.shape[1]!=len(self.spectator_modes)
                or rows.dtype.kind not in 'iu' or np.any(rows<0)):
            raise ValueError('nonnegative integer spectator occupation rows required')
        result=[]
        for row in rows:
            key=tuple(map(int,row))
            if key not in self._cache:
                doorway=self.conditional_doorways(row[None,:])
                target=self.initial_energy_cm-doorway['spectator_energies_cm']
                value=self.prepared_spectrum.evaluate(doorway['source_hartree'],target,
                    sigma_cm=self.sigma,tolerance_per_strength=self.uniform_density_tolerance,workers=self.workers)
                if not value['accepted'][0]:raise ValueError('prepared spectrum does not enclose the conditional target')
                self._cache[key]=float(value['densities_per_cm'][0])
                if len(self._cache)>self.maximum_cached_states:self._cache.popitem(last=False)
            result.append(self._cache[key])
        return np.asarray(result)
