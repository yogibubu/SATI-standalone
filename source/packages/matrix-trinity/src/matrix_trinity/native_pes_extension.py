"""Direct native-PES enlargement of an immutable modal Hamiltonian by one mode."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg.blas import dsyrk,dgemm
from scipy.sparse import csr_matrix
from scipy.special import roots_hermite

from .vibronic_sparse import SparseVibrationalHamiltonian


def native_core_quadrature_coordinates(core):
    """Coordinates of the parent's tensor GH rule, in its original row order."""
    count = core.diagnostics['quadrature_points_per_axis']
    if not isinstance(count, int) or count < 2:
        raise ValueError('parent must use a declared isotropic direct-native quadrature')
    axis = roots_hermite(count)[0]
    indices = np.array(np.unravel_index(np.arange(count**len(core.frequencies_cm)),
                                       (count,)*len(core.frequencies_cm))).T
    return axis[indices]


def harmonic_reference_for_native_extension(core, native):
    """Recover the unchanged harmonic parent blocks of a saved native extension.

    Reuses the original basis-order and block formula. No potential evaluation
    or quadrature projection is needed to resume after spectral acquisition
    failed. Metadata and every expected retained row are checked before reuse.
    """
    from scipy.sparse import block_diag, diags
    if not isinstance(core, SparseVibrationalHamiltonian) or not isinstance(native, SparseVibrationalHamiltonian):
        raise TypeError('immutable parent and native Hamiltonians required')
    d = native.diagnostics
    if (d.get('parent_sha256') != core.fingerprint
            or d.get('potential_identifier') != core.diagnostics.get('potential_identifier')
            or native.basis.shape[1] != core.basis.shape[1]+1
            or not np.array_equal(native.frequencies_cm[:-1], core.frequencies_cm)):
        raise ValueError('saved extension must belong to the identical parent and field')
    cap, w, ceiling = d['maximum_extra_quanta'], d['extra_frequency_cm'], d['maximum_harmonic_energy_cm']
    if (not isinstance(cap, int) or cap < 1 or not np.isfinite(w) or w <= 0
            or w != native.frequencies_cm[-1] or not np.isfinite(ceiling) or ceiling <= 0):
        raise ValueError('valid saved extension occupation, frequency and energy limits required')
    energy = core.basis@core.frequencies_cm
    start = 0; blocks = []
    for occupation in range(cap+1):
        take = np.flatnonzero(energy+occupation*w <= ceiling+1e-10)
        if not len(take):continue
        stop = start+len(take)
        expected = np.column_stack([core.basis[take], np.full(len(take), occupation)])
        if not np.array_equal(native.basis[start:stop], expected):
            raise ValueError('saved extension basis differs from the declared parent blocks')
        blocks.append(core.matrix_cm[take][:, take]+diags(np.full(len(take), (occupation+.5)*w)))
        start = stop
    if start != len(native.basis):
        raise ValueError('saved extension contains unexpected basis rows')
    matrix = block_diag(blocks, format='csr')
    matrix.eliminate_zeros(); matrix.sort_indices()
    return SparseVibrationalHamiltonian(native.basis, native.frequencies_cm, matrix,
        dict(d, extra_mode_model='harmonic'))


def build_native_mode_extension(core, extra_frequency_cm, potential_evaluator, *,
        core_potential_cm, maximum_extra_quanta, extra_quadrature_points,
        maximum_harmonic_energy_cm, potential_identifier,
        extra_coordinate_identifier,
        maximum_states=24000, maximum_grid_points=2000000,
        maximum_workspace_bytes=64*1024**3, batch_size=128,
        residual_cache_directory=None, progress=None, prepared_residual_checkpoint=None):
    """Return matched harmonic-extension and full-native finite Hamiltonians.

    Both have the same complete declared occupation set. The reference is the
    exact parent matrix plus a harmonic extra coordinate. The native correction
    is the direct quadrature of V(A,b)-V(A,0)-w_b*b**2/2. No derivatives, fitted
    expansion, or coupling pruning enter. Parent and extra-mode basis/quadrature
    convergence, and simultaneous enlargement by multiple modes, remain separate.

    Symmetric rank updates project the residual without a dense temporary matrix
    for each grid batch. Optional residual checkpoints permit PES acquisition to
    resume independently of matrix assembly.
    A prepared checkpoint (directory, source Hamiltonian diagnostics) may seed
    a different occupation cutoff only on the identical identified tensor grid.
    Its original metadata and parent values must reproduce its saved cache key.
    """
    if not isinstance(core, SparseVibrationalHamiltonian):
        raise TypeError('immutable parent Hamiltonian required')
    if prepared_residual_checkpoint is not None and residual_cache_directory is None:
        raise ValueError('a destination cache is required for a prepared residual checkpoint')
    w = float(extra_frequency_cm)
    if (not np.isfinite(w) or w <= 0 or not callable(potential_evaluator)
            or potential_identifier != core.diagnostics.get('potential_identifier')
            or not isinstance(extra_coordinate_identifier,str) or not extra_coordinate_identifier
            or not np.isfinite(maximum_harmonic_energy_cm) or maximum_harmonic_energy_cm <= 0):
        raise ValueError('positive frequency/energy and identical identified native potential required')
    for value in (maximum_extra_quanta, extra_quadrature_points, maximum_states,
                  maximum_grid_points, maximum_workspace_bytes, batch_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('positive integer basis, quadrature and resource controls required')
    if extra_quadrature_points <= maximum_extra_quanta+1:
        raise ValueError('extra quadrature must represent retained HO products and quadratic reference')
    ncore = len(core.frequencies_cm)
    core_count = core.diagnostics['quadrature_points_per_axis']
    core_coordinates = native_core_quadrature_coordinates(core)
    vcore = np.asarray(core_potential_cm)
    if np.iscomplexobj(vcore) or vcore.shape != (len(core_coordinates),) or not np.all(np.isfinite(vcore)):
        raise ValueError('one finite real parent-potential value per original quadrature node required')
    rows=[]; parent_indices=[]; blocks=[]
    core_energy=core.basis@core.frequencies_cm
    for occupation in range(maximum_extra_quanta+1):
        take=np.flatnonzero(core_energy+occupation*w <= maximum_harmonic_energy_cm+1e-10)
        if not len(take):continue
        start=len(rows)
        rows.extend(np.column_stack([core.basis[take],np.full(len(take),occupation)]).tolist())
        parent_indices.extend(take.tolist());blocks.append((start,len(rows),occupation,take))
    basis=np.asarray(rows,dtype=np.int64);size=len(basis)
    points=len(core_coordinates)*extra_quadrature_points
    workspace=64*size**2+48*batch_size*size+32*points
    if size>maximum_states or points>maximum_grid_points or workspace>maximum_workspace_bytes:
        raise MemoryError(f'native extension exceeds declared resources: {size} states, {points} points, {workspace} bytes')
    # Amortize full matrix updates over larger BLAS ranks when the same declared
    # workspace permits it. Acquisition batching and its cache identity remain
    # unchanged, so already acquired potentials can be reused bit for bit.
    projection_batch_size=max(batch_size,min(512,
        (maximum_workspace_bytes-64*size**2-32*points)//(48*size)))
    projection_workspace=64*size**2+48*projection_batch_size*size+32*points
    counts=[core_count]*ncore+[extra_quadrature_points]
    axes=[];functions=[];gram_errors=[]
    for mode,count in enumerate(counts):
        axis,weights=roots_hermite(count);axes.append(axis)
        maximum=int(basis[:,mode].max())
        if count<=maximum+1:raise ValueError('parent quadrature does not represent the extension basis')
        poly=np.empty((maximum+1,count));poly[0]=np.pi**(-.25)
        if maximum:poly[1]=np.sqrt(2)*axis*poly[0]
        for order in range(2,maximum+1):
            poly[order]=np.sqrt(2/order)*axis*poly[order-1]-np.sqrt((order-1)/order)*poly[order-2]
        f=poly*np.sqrt(weights);functions.append(f)
        gram_errors.append(float(np.max(abs(f@f.T-np.eye(maximum+1)))))
    if max(gram_errors)>1e-10:raise ArithmeticError('extension GH orthogonality failed')
    reference=np.zeros((size,size),order='F')
    for start,stop,occupation,take in blocks:
        reference[start:stop,start:stop]=core.matrix_cm[take][:,take].toarray()
        diagonal=np.arange(start,stop);reference[diagonal,diagonal]+=(occupation+.5)*w
    metadata=dict(schema='matrix.trinity.native_mode_extension.v1',parent_sha256=core.fingerprint,
        potential_identifier=potential_identifier,maximum_extra_quanta=maximum_extra_quanta,
        extra_frequency_cm=w,extra_coordinate_identifier=extra_coordinate_identifier,
        maximum_harmonic_energy_cm=maximum_harmonic_energy_cm,
        quadrature_points_per_axis=counts,basis_size=size,quadrature_point_count=points,
        workspace_guard_bytes=workspace,quadrature_orthogonality_errors=gram_errors,
        complete_native_functions=True,polynomial_reexpansion_used=False,coupling_pruning_used=False,
        full_molecular_convergence_certified=False)
    reference_h=SparseVibrationalHamiltonian(basis,np.r_[core.frequencies_cm,w],csr_matrix(reference),
        dict(metadata,extra_mode_model='harmonic'))
    digest=hashlib.sha256()
    digest.update(json.dumps(metadata,sort_keys=True).encode())
    digest.update(np.ascontiguousarray(vcore).tobytes())
    cache_key=digest.hexdigest();acquired=0;started=time.monotonic()
    cache=Path(residual_cache_directory) if residual_cache_directory is not None else None
    if cache is not None:
        cache.mkdir(parents=True,exist_ok=True);status=cache/'acquisition.json';values=cache/'residual.npy'
        if prepared_residual_checkpoint is not None and not status.exists():
            source_directory,source_diagnostics=prepared_residual_checkpoint
            source_directory=Path(source_directory)
            source_metadata={key:source_diagnostics[key] for key in metadata}
            physical_keys=('schema','parent_sha256','potential_identifier','extra_frequency_cm',
                'extra_coordinate_identifier','quadrature_points_per_axis','quadrature_point_count')
            if any(source_metadata[key]!=metadata[key] for key in physical_keys):
                raise ValueError('prepared residual checkpoint uses a different potential or tensor grid')
            source_digest=hashlib.sha256()
            source_digest.update(json.dumps(source_metadata,sort_keys=True).encode())
            source_digest.update(np.ascontiguousarray(vcore).tobytes())
            source_key=source_digest.hexdigest()
            saved=json.loads((source_directory/'acquisition.json').read_text())
            if (saved['cache_key']!=source_key
                    or source_diagnostics.get('residual_cache_key')!=source_key):
                raise ValueError('prepared residual checkpoint identity cannot be verified')
            count=saved['acquired_points']
            source=np.load(source_directory/'residual.npy',mmap_mode='r',allow_pickle=False)
            if (isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=points
                    or source.shape!=(points,) or source.dtype!=np.dtype(float)
                    or not np.all(np.isfinite(source[:count]))):
                raise ValueError('invalid prepared residual checkpoint prefix')
            copied=np.lib.format.open_memmap(values,mode='w+',dtype=float,shape=(points,))
            copied[:count]=source[:count];copied.flush();del copied
            temporary=status.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(cache_key=cache_key,acquired_points=count,
                total_points=points,copied_from_cache_key=source_key))+'\n')
            temporary.replace(status)
            if progress:progress(dict(stage='reused potential checkpoint',acquired_points=count,
                total_points=points,source_cache_key=source_key))
        if status.exists():
            saved=json.loads(status.read_text())
            if saved['cache_key']!=cache_key:raise ValueError('cached PES residual belongs to another calculation')
            acquired=saved['acquired_points'];residual=np.lib.format.open_memmap(values,mode='r+')
            if residual.shape!=(points,) or not 0<=acquired<=points:raise ValueError('invalid residual checkpoint')
        else:residual=np.lib.format.open_memmap(values,mode='w+',dtype=float,shape=(points,))
    else:residual=np.empty(points)
    for start in range(acquired,points,batch_size):
        flat=np.arange(start,min(start+batch_size,points));indices=np.array(np.unravel_index(flat,tuple(counts))).T
        coordinates=np.column_stack([axis[indices[:,i]] for i,axis in enumerate(axes)])
        values=np.asarray(potential_evaluator(coordinates))
        if np.iscomplexobj(values) or values.shape!=(len(flat),) or not np.all(np.isfinite(values)):
            raise ValueError('native PES must return one finite real energy per geometry')
        residual[flat]=values-vcore[flat//extra_quadrature_points]-.5*w*coordinates[:,-1]**2
        acquired=int(flat[-1])+1
        if acquired%2048==0 or acquired==points:
            if cache is not None:
                residual.flush();temporary=status.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(cache_key=cache_key,acquired_points=acquired,total_points=points))+'\n')
                temporary.replace(status)
            if progress:progress(dict(stage='potential',acquired_points=acquired,total_points=points,
                                      elapsed_seconds=time.monotonic()-started))
    matrix=reference;del reference
    # Integrate the extra coordinate before the expensive core contractions.
    # This is exactly the same tensor quadrature, without repeating a core
    # matrix update for each extra-coordinate node.
    from .native_tensor_projection import (plan_native_tensor_projection,
                                            project_native_tensor_correction)
    tensor_plan=plan_native_tensor_projection(functions[:-1],len(core.basis),
        reserved_bytes=workspace,maximum_workspace_bytes=maximum_workspace_bytes)
    if tensor_plan is not None:
        projection_started=time.monotonic()
        test_columns=np.linspace(0,size-1,min(32,size),dtype=int)
        before=matrix[np.ix_(test_columns,test_columns)].copy()
        project_native_tensor_correction(matrix,core.basis,blocks,functions,residual,tensor_plan,progress=progress)
        # Independent direct quadrature on distributed retained rows checks the
        # complete contraction, not just the first acquisition batch.
        check=np.zeros_like(before)
        for start in range(0,len(core_coordinates),512):
            flat=np.arange(start,min(start+512,len(core_coordinates)))
            indices=np.array(np.unravel_index(flat,(core_count,)*ncore)).T
            pc=np.ones((len(flat),len(test_columns)))
            for mode,f in enumerate(functions[:-1]):
                pc*=f[basis[test_columns,mode][None,:],indices[:,mode,None]]
            full=(pc[:,None,:]*functions[-1][basis[test_columns,-1]].T[None,:,:]).reshape(-1,len(test_columns))
            values=np.asarray(residual).reshape(-1,extra_quadrature_points)[flat].ravel()
            check+=full.T@(values[:,None]*full)
        # Only the upper off-diagonal blocks have been filled so far.
        obtained=matrix[np.ix_(test_columns,test_columns)]
        obtained=np.triu(obtained)+np.triu(obtained,1).T
        maximum_rank_update_check=float(np.max(abs(obtained-before-check)))
        if maximum_rank_update_check>1e-10*max(1.,float(np.max(abs(check)))):
            raise ArithmeticError('tensor contraction differs from direct tensor quadrature')
        projection_seconds=time.monotonic()-projection_started
    else:
        block_matrices={}
        for i,(si,ei,oi,ti) in enumerate(blocks):
            for j,(sj,ej,oj,tj) in enumerate(blocks[i:],i):
                block_matrices[i,j]=np.array(matrix[si:ei,sj:ej],order='F',copy=True)
        test_columns=np.linspace(0,size-1,min(32,size),dtype=int)
        test_blocks=np.searchsorted([b[1] for b in blocks],test_columns,side='right')
        def sample_blocks():
            result=np.empty((len(test_columns),len(test_columns)))
            for a,(column,bi) in enumerate(zip(test_columns,test_blocks)):
                li=column-blocks[bi][0]
                for b,(other,bj) in enumerate(zip(test_columns,test_blocks)):
                    lj=other-blocks[bj][0]
                    if bi<bj:result[a,b]=block_matrices[bi,bj][li,lj]
                    elif bi>bj:result[a,b]=block_matrices[bj,bi][lj,li]
                    else:result[a,b]=block_matrices[bi,bi][min(li,lj),max(li,lj)]
            return result
        extra_products=functions[-1][:,None,:]*functions[-1][None,:,:]
        maximum_rank_update_check=0.;projection_started=time.monotonic()
        for start in range(0,len(core_coordinates),projection_batch_size):
            flat=np.arange(start,min(start+projection_batch_size,len(core_coordinates)))
            indices=np.array(np.unravel_index(flat,(core_count,)*ncore)).T
            projection=np.ones((len(flat),len(core.basis)))
            for mode,f in enumerate(functions[:-1]):
                projection*=f[core.basis[:,mode][None,:],indices[:,mode,None]]
            values=np.asarray(residual).reshape(-1,extra_quadrature_points)[flat]
            collapsed=(values@extra_products.reshape(-1,extra_quadrature_points).T).reshape(
                len(flat),len(functions[-1]),len(functions[-1]))
            if start==0:
                pc=projection[:,np.asarray(parent_indices)[test_columns]]
                aux=basis[test_columns,-1]
                full=(pc[:,None,:]*functions[-1][aux].T[None,:,:]).reshape(-1,len(aux))
                check=full.T@(values.ravel()[:,None]*full);before=sample_blocks()
            for i,(_,_,oi,ti) in enumerate(blocks):
                for j,(_,_,oj,tj) in enumerate(blocks[i:],i):
                    weights=collapsed[:,oi,oj]
                    for sign,selected in [(1.,weights>0),(-1.,weights<0)]:
                        if not np.any(selected):continue
                        root=np.sqrt(abs(weights[selected,None]))
                        left=np.asfortranarray(projection[selected][:,ti]*root)
                        if i==j:
                            block_matrices[i,j]=dsyrk(sign,left,beta=1.,c=block_matrices[i,j],
                                trans=1,lower=0,overwrite_c=1)
                        else:
                            right=np.asfortranarray(projection[selected][:,tj]*root)
                            block_matrices[i,j]=dgemm(sign,left,right,beta=1.,c=block_matrices[i,j],
                                trans_a=1,overwrite_c=1)
            if start==0:
                maximum_rank_update_check=float(np.max(abs(sample_blocks()-before-check)))
                if maximum_rank_update_check>1e-10*max(1.,float(np.max(abs(check)))):
                    raise ArithmeticError('collapsed BLAS projection differs from direct tensor quadrature')
            if (int(flat[-1])+1)%2048==0 or int(flat[-1])+1==len(core_coordinates):
                if progress:progress(dict(stage='projection',projected_core_points=int(flat[-1])+1,
                    total_core_points=len(core_coordinates),elapsed_seconds=time.monotonic()-started))
        projection_seconds=time.monotonic()-projection_started
        for (i,j),block in block_matrices.items():
            si,ei,_,_=blocks[i];sj,ej,_,_=blocks[j]
            matrix[si:ei,sj:ej]=block
        del block_matrices
    # Fill the untouched triangle in blocks without another complete dense array.
    for start in range(0,size,256):
        stop=min(start+256,size)
        matrix[stop:,start:stop]=matrix[start:stop,stop:].T
        upper=np.triu(matrix[start:stop,start:stop])
        matrix[start:stop,start:stop]=upper+np.triu(upper,1).T
    native_h=SparseVibrationalHamiltonian(basis,np.r_[core.frequencies_cm,w],csr_matrix(matrix),
        dict(metadata,extra_mode_model='full native',rank_update_check_maximum_cm=maximum_rank_update_check,
             projection_algorithm=('symmetric univariate pair tensor contractions' if tensor_plan is not None
                                   else 'extra-mode contraction then core BLAS blocks'),
             tensor_projection_plan=tensor_plan,
             projection_core_batch_size=None if tensor_plan is not None else projection_batch_size,
             projection_workspace_guard_bytes=(tensor_plan['workspace_guard_bytes'] if tensor_plan is not None
                                               else projection_workspace),
             projection_seconds=projection_seconds,
             residual_cache_key=cache_key,elapsed_seconds=time.monotonic()-started))
    return reference_h,native_h


def acquire_harmonic_extension_spectrum(hamiltonian, *, target_max_cm,
        gaussian_sigma_cm, tolerance_per_strength=1e-18, workers=4,
        maximum_workspace_bytes=64*1024**3,parent=None,parent_spectrum=None):
    """Acquire an exact harmonic-extension window through its independent blocks."""
    from .vibronic_spectral_cache import (acquire_gaussian_spectrum,PreparedGaussianSpectrum,
        PreparedBlockGaussianSpectrum)
    if hamiltonian.diagnostics.get('extra_mode_model')!='harmonic':
        raise ValueError('identified harmonic extension required')
    last=hamiltonian.basis[:,-1];coo=hamiltonian.matrix_cm.tocoo()
    if np.any((last[coo.row]!=last[coo.col])&(coo.data!=0)):
        raise ValueError('harmonic extra-coordinate occupation must be conserved exactly')
    started=time.monotonic();parts=[];omitted=[];diagnostics=[]
    if (parent is None)!=(parent_spectrum is None):raise ValueError('parent matrix and spectrum must be supplied together')
    if parent is not None and (parent_spectrum.hamiltonian_sha256!=parent.fingerprint
            or hamiltonian.diagnostics['parent_sha256']!=parent.fingerprint):
        raise ValueError('immutable parent spectral identity mismatch')
    peak=1/(np.sqrt(2*np.pi)*gaussian_sigma_cm)
    ceiling=target_max_cm+gaussian_sigma_cm*np.sqrt(2*max(0.,np.log(2*peak/tolerance_per_strength)))
    for occupation in np.unique(last):
        take=np.flatnonzero(last==occupation)
        block=SparseVibrationalHamiltonian(hamiltonian.basis[take],hamiltonian.frequencies_cm,
            hamiltonian.matrix_cm[take][:,take],hamiltonian.diagnostics)
        if parent is not None:
            if not np.array_equal(block.basis[:,:-1],parent.basis):
                raise ValueError('spectral parent reuse requires an unpruned product basis')
            shift=(occupation+.5)*hamiltonian.frequencies_cm[-1]
            from scipy.sparse import eye
            difference=block.matrix_cm-parent.matrix_cm-shift*eye(len(parent.basis),format='csr')
            if np.max(abs(difference.data),initial=0)>1e-9:
                raise ValueError('harmonic extension block is not the identical shifted parent')
            if (parent_spectrum.omitted_lower_bound_cm is not None
                    and parent_spectrum.omitted_lower_bound_cm+shift<ceiling):
                raise ValueError('cached parent eigenwindow is insufficient for the extension')
            if parent_spectrum.omitted_lower_bound_cm is not None:
                omitted.append(parent_spectrum.omitted_lower_bound_cm+shift)
            sources=(parent_spectrum.parts if isinstance(parent_spectrum,PreparedBlockGaussianSpectrum)
                     else ((np.arange(len(parent.basis)),parent_spectrum),))
            for source_rows,source in sources:
                levels=source.energies_cm+shift;count=int(np.searchsorted(levels,ceiling,side='right'))
                if count<len(levels):omitted.append(float(levels[count]))
                if not count:continue
                # Ordered lowest levels form a prefix: retain read-only views
                # instead of copying the common parent eigenvectors.
                identity=(block.fingerprint if len(sources)==1 else hashlib.sha256(
                    (block.fingerprint+source.hamiltonian_sha256).encode()+source_rows.tobytes()).hexdigest())
                subset=PreparedGaussianSpectrum(levels[:count],source.eigenvectors[:,:count],
                    min(omitted) if omitted else None,identity,
                    dict(source.diagnostics,harmonic_shift_cm=shift,parent_spectrum_reused=True))
                parts.append((take[source_rows],subset));diagnostics.append(subset.diagnostics)
            continue
        # A very high harmonic block may have no levels in the requested window.
        # Its exact minimum is found without constructing its full eigenwindow.
        from scipy.linalg import eigh
        minimum=eigh(block.matrix_cm.toarray(),subset_by_index=(0,0),eigvals_only=True)[0]
        if minimum>ceiling:
            omitted.append(float(minimum));continue
        spectrum=acquire_gaussian_spectrum(block,target_max_cm=target_max_cm,
            gaussian_sigma_cm=gaussian_sigma_cm,tolerance_per_strength=tolerance_per_strength,
            workers=workers,maximum_workspace_bytes=maximum_workspace_bytes)
        parts.append((take,spectrum));diagnostics.append(spectrum.diagnostics)
        if spectrum.omitted_lower_bound_cm is not None:omitted.append(spectrum.omitted_lower_bound_cm)
    if not parts:raise ValueError('requested harmonic extension window contains no levels')
    level_count=sum(len(s.energies_cm) for _,s in parts)
    required=sum(3*s.eigenvectors.nbytes+rows.nbytes for rows,s in parts)
    if required>maximum_workspace_bytes:
        raise MemoryError('combined block eigenwindow exceeds workspace budget')
    return PreparedBlockGaussianSpectrum(len(last),tuple(parts),min(omitted) if omitted else None,
        hamiltonian.fingerprint,dict(acquisition_method='exact conserved-occupation blocks',
            block_diagnostics=diagnostics,level_count=level_count,acquisition_seconds=time.monotonic()-started,
            spectral_storage='disjoint_row_blocks.v1',spectral_workspace_guard_bytes=required,
            cross_block_orthogonality='exact disjoint row support'))
