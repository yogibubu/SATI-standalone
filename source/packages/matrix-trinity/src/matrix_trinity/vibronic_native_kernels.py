"""Typed C++ adaptive kernels sharing the existing NumPy checkpoint arrays."""
import ctypes as ct
from pathlib import Path
import tempfile
import threading
import numpy as np

P=ct.c_void_p
S=ct.c_size_t
I=ct.c_int64
U=ct.c_uint64


class TreeRequest(ct.Structure):
    _fields_=[(name,S) for name in ('dimension','length','budget','capacity')]+[
        (name,P) for name in ('costs','values','parents','original','ancestors','axes','ranks')]+[
        ('allocated',S),('visited',S),('heap_cost',P),('heap_node',P),('heap_axis',P)]+[
        (name,S) for name in ('heap_size','modes','qi','qf','maximum_class')]


class PackedRequest(ct.Structure):
    _fields_=[('states',P),('rows',S),('dimension',S),('keys',P),('words',S),('maximum_integrals',S),
        ('slots',P),('slot_count',S),('values',P),('count',S),('edge_count',S),
        ('matrix',P),('linear',P),('ht',P),('bits',ct.c_uint),('mask',U),('maximum_depth',S),
        ('ids',P),('completed',S),('exhausted',ct.c_int)]


class OperatorRequest(ct.Structure):
    _fields_=[(name,S) for name in ('rows','dimension','channels','forms','count')]+[
        (name,P) for name in ('states','ids','values','constant','linear','weights','q','strengths','norms')]


def pointer(array):
    return P(array.ctypes.data)


class NativeKernels:
    """Explicitly loaded library; no compilation, binary search or implicit fallback."""
    def __init__(self,path,*,recurrence_mode='direct',recurrence_workers=1):
        if recurrence_mode not in {'direct','levels'}:raise ValueError('Recurrence mode must be direct or levels')
        if not isinstance(recurrence_workers,int) or not 1<=recurrence_workers<=32:
            raise ValueError('Recurrence workers must be an integer from 1 to 32')
        self.recurrence_mode=recurrence_mode;self.recurrence_workers=recurrence_workers
        self.path=str(Path(path).resolve());self.api=ct.CDLL(self.path)
        signatures={
            'matrix_prepare_dipole_gaussian':([ct.c_char_p,S]+[P]*5,ct.c_int),
            'matrix_dipole_amplitudes':([S,S]+[P]*7,ct.c_int),
            'matrix_fill_occupations':([S]*4+[P]*5,ct.c_int),
            'matrix_dipole_correlation':([ct.c_char_p,S,P,ct.c_double,P,P,P],ct.c_int),
            'matrix_dipole_fft':([S,ct.c_double,P,P],ct.c_int),
            'matrix_rate_moments':([S,P,P,ct.c_double,ct.c_double,P,P,P],ct.c_int),
            'matrix_overlap_grid':([S,S,P,P,P,P,ct.c_double,P],ct.c_int),
            'matrix_operator_rows':([S,S,S]+[P]*6,ct.c_int),
            'matrix_contract_orders':([S,S]+[P]*5,ct.c_int),
            'matrix_dipole_ti_batch':([ct.POINTER(PackedRequest),P,S,S]+[P]*7+
                [ct.c_double]*3+[ct.c_int]+[P]*7,ct.c_int),
            'matrix_prepare_gaussian_geometry':([ct.c_char_p,S]+[P]*4,ct.c_int),
            'matrix_insert_dipole':([S]+[P]*5,ct.c_int),
            'matrix_selection_error':([],ct.c_char_p),
            'matrix_selection_abi_sizes':([ct.POINTER(S)],None),
            'matrix_tree_create':([ct.POINTER(TreeRequest)],P),
            'matrix_tree_destroy':([P],None),'matrix_tree_heap_size':([P],S),
            'matrix_tree_batch':([P,S,P,ct.POINTER(S),ct.POINTER(S)],I),
            'matrix_tree_export_heap':([P,P,P,P],ct.c_int),
            'matrix_packed_evaluate':([ct.POINTER(PackedRequest)],ct.c_int),
            'matrix_packed_rehash':([P,S,S,P,S],ct.c_int),
            'matrix_operator_strengths':([ct.POINTER(OperatorRequest)],ct.c_int),
            'matrix_prepare_operator_gaussian':([ct.c_char_p,S,S,S]+[P]*8,ct.c_int),
            'matrix_gaussian_marginal_1':([ct.c_double,ct.c_double,ct.c_double,S,P],ct.c_int),
            'matrix_gaussian_marginal_2':([P,P,P,S,S,S,S,P],ct.c_int),
        }
        for name,(arguments,result) in signatures.items():
            function=getattr(self.api,name);function.argtypes=arguments;function.restype=result
        # The unchanged ABI remains usable for explicit reference comparisons.
        self.persistent_recurrence=hasattr(self.api,'matrix_packed_context_create')
        if self.persistent_recurrence:
            context_signatures={
                'matrix_packed_context_create':([ct.c_int,S],P),
                'matrix_packed_context_destroy':([P],None),
                'matrix_packed_context_evaluate':([P,ct.POINTER(PackedRequest)],ct.c_int),
                'matrix_packed_context_stats':([P,P],ct.c_int),
                'matrix_dipole_ti_batch_context':(signatures['matrix_dipole_ti_batch'][0]+[P],ct.c_int),
            }
            for name,(arguments,result) in context_signatures.items():
                function=getattr(self.api,name);function.argtypes=arguments;function.restype=result
        elif recurrence_mode!='direct' or recurrence_workers!=1:
            raise RuntimeError('This native library does not implement level recurrence workspaces')
        sizes=(S*4)();self.api.matrix_selection_abi_sizes(sizes)
        if list(sizes)!=[ct.sizeof(TreeRequest),ct.sizeof(PackedRequest),ct.sizeof(OperatorRequest),16]:
            raise RuntimeError('Native kernel ABI does not match the typed adapter')

    def check(self,result):
        if result<0:
            message=self.api.matrix_selection_error().decode()
            if 'budget' in message.lower():raise MemoryError(message)
            raise RuntimeError(message)

    def packed_batch(self,cache,states):
        states=np.ascontiguousarray(states,dtype=np.int64);rows,d=states.shape
        ids=np.empty((rows,d+1),np.int64);live=np.ascontiguousarray(cache.ht,dtype=np.uint8)
        depth=int(states.sum(axis=1).max())+2 if rows else 2
        request=PackedRequest(pointer(states),rows,d,pointer(cache.keys),cache.keys.shape[1],len(cache.values),
            pointer(cache.slots),len(cache.slots),pointer(cache.values),cache.count,cache.edge_count,
            pointer(cache.matrix),pointer(cache.linear),pointer(live),cache.bits,int(cache.mask),depth,
            pointer(ids),0,0)
        context=self.packed_context(cache)
        if context is None:result=self.api.matrix_packed_evaluate(ct.byref(request))
        else:
            with context.lock:result=self.api.matrix_packed_context_evaluate(context.handle,ct.byref(request))
        cache.count=request.count;cache.edge_count=request.edge_count;self.check(result)
        return ids[:request.completed],bool(request.exhausted)

    def packed_context(self,cache):
        if not self.persistent_recurrence:return None
        context=getattr(cache,'_native_workspace',None)
        if context is None or not context.handle or context.library is not self:
            if context is not None:context.close()
            context=NativePackedWorkspace(self);cache._native_workspace=context
        return context

    def workspace_stats(self,cache):
        context=getattr(cache,'_native_workspace',None)
        if context is None:return None
        values=np.empty(9,np.uintp)
        with context.lock:self.check(self.api.matrix_packed_context_stats(context.handle,pointer(values)))
        names=('batches','lookups','tag_rejections','key_comparisons','buffer_growths','index_bytes',
            'level_nodes','parallel_levels','workers')
        return dict(mode=self.recurrence_mode,**dict(zip(names,map(int,values))))

    def rehash(self,cache):
        context=getattr(cache,'_native_workspace',None)
        if context is not None:context.close();cache._native_workspace=None
        self.check(self.api.matrix_packed_rehash(pointer(cache.keys),cache.keys.shape[1],cache.count,
            pointer(cache.slots),len(cache.slots)))

    def marginal1(self,vx,vp,displacement,maximum):
        output=np.empty(maximum+1)
        self.check(self.api.matrix_gaussian_marginal_1(vx,vp,displacement,maximum,pointer(output)))
        return output

    def marginal2(self,vx,vp,mean,first,second):
        vx=np.ascontiguousarray(vx,dtype=np.float64);vp=np.ascontiguousarray(vp,dtype=np.float64)
        mean=np.ascontiguousarray(mean,dtype=np.float64);output=np.empty((first+1,second+1))
        if (first+1)**2*(second+1)**2>100000:
            from scipy.fft import next_fast_len
            n0=next_fast_len(8*(first+1)) if first else 1;n1=next_fast_len(8*(second+1)) if second else 1
        else:n0=n1=1
        self.check(self.api.matrix_gaussian_marginal_2(pointer(vx),pointer(vp),pointer(mean),
            first,second,n0,n1,pointer(output)))
        return output

    def strengths(self,states,ids,cache,constant,linear,forms,q):
        states=np.ascontiguousarray(states,dtype=np.int64);ids=np.ascontiguousarray(ids,dtype=np.int64)
        constant=np.ascontiguousarray(constant,dtype=np.complex128);linear=np.ascontiguousarray(linear,dtype=np.complex128)
        forms=np.ascontiguousarray(forms,dtype=np.complex128);q=np.ascontiguousarray(q,dtype=np.float64)
        rows,d=states.shape;strengths=np.empty((len(forms),rows));norms=np.empty(rows)
        request=OperatorRequest(rows,d,len(constant),len(forms),cache.count,pointer(states),pointer(ids),
            pointer(cache.values),pointer(constant),pointer(linear),pointer(forms),pointer(q),pointer(strengths),pointer(norms))
        self.check(self.api.matrix_operator_strengths(ct.byref(request)));return strengths,norms

    def occupations(self,states,totals,classes,offsets,binomial):
        arrays=[np.ascontiguousarray(a,dtype=np.int64) for a in (totals,classes,offsets,binomial)]
        self.check(self.api.matrix_fill_occupations(len(states),states.shape[1],len(totals),binomial.shape[1],
            *[pointer(a) for a in arrays],pointer(states)))

    def amplitudes(self,states,ids,values,mu,ht0,htlinear):
        states=np.ascontiguousarray(states,dtype=np.int64);ids=np.ascontiguousarray(ids,dtype=np.int64)
        values=np.ascontiguousarray(values,dtype=np.float64)
        mu=np.ascontiguousarray(mu,dtype=np.complex128);ht0=np.ascontiguousarray(ht0,dtype=np.complex128)
        htlinear=np.ascontiguousarray(htlinear,dtype=np.complex128);output=np.empty((len(states),5))
        self.check(self.api.matrix_dipole_amplitudes(len(states),states.shape[1],*[pointer(a) for a in
            (states,ids,values,mu,ht0,htlinear,output)]))
        return output

    def prepare_dipole(self,model,temperature_K):
        from .vibronic_native import write_native_harmonic_input
        n=len(model.initial_frequencies_cm);d=2*n
        matrix=np.empty((d,d));linear=np.empty(d);log_ground=np.empty(1)
        constant=np.empty(3,complex);insertion=np.empty((3,d),complex)
        with tempfile.TemporaryDirectory(prefix='matrix-native-dipole-') as folder:
            path=Path(folder)/'model.dat'
            write_native_harmonic_input(model,path,method='TI',temperature_K=temperature_K,
                maximum_initial_quanta=0,maximum_final_quanta=0,maximum_class=0)
            self.check(self.api.matrix_prepare_dipole_gaussian(str(path).encode(),n,*[pointer(a) for a in
                (matrix,linear,log_ground,constant,insertion)]))
        return dict(matrix=matrix,linear=linear,log_ground=float(log_ground[0]),ht_constant=constant,ht_linear=insertion)

    def correlation(self,model,times,temperature_K,direction):
        from .vibronic_native import write_native_harmonic_input
        times=np.ascontiguousarray(times,dtype=np.float64);output=np.empty((3,len(times)),complex)
        bound=np.empty(1);certified=np.empty(1,np.int32)
        with tempfile.TemporaryDirectory(prefix='matrix-native-correlation-') as folder:
            path=Path(folder)/'model.dat'
            write_native_harmonic_input(model,path,method='TD',temperature_K=temperature_K)
            self.check(self.api.matrix_dipole_correlation(str(path).encode(),len(times),pointer(times),direction,
                pointer(output),pointer(bound),pointer(certified)))
        return output,dict(certified_LU_logdet=bool(certified[0]),logdet_absolute_bound=float(bound[0]))

    def fft(self,signal,spacing):
        signal=np.ascontiguousarray(signal,dtype=np.complex128);points=signal.shape[1]
        output=np.empty((3,2*(points-1)))
        self.check(self.api.matrix_dipole_fft(points,spacing,pointer(signal),pointer(output)))
        return output

    def rate_moments(self,energies,strengths,sigma,gamma):
        energies=np.ascontiguousarray(energies,dtype=np.float64)
        strengths=np.ascontiguousarray(strengths,dtype=np.float64)
        if energies.shape!=strengths.shape or energies.ndim!=1:raise ValueError('Equal energy/strength vectors required')
        einstein=np.empty(1);density=np.empty(1);peak=np.empty(1)
        self.check(self.api.matrix_rate_moments(len(energies),pointer(energies),pointer(strengths),sigma,gamma,
            pointer(einstein),pointer(density),pointer(peak)))
        return float(einstein[0]),float(density[0]),float(peak[0])

    def overlap_grid(self,matrix,linear,dimensions,strides,ground):
        matrix=np.ascontiguousarray(matrix,dtype=np.float64);linear=np.ascontiguousarray(linear,dtype=np.float64)
        dimensions=np.ascontiguousarray(dimensions,dtype=np.int64);strides=np.ascontiguousarray(strides,dtype=np.int64)
        from math import prod
        entries=prod(map(int,dimensions));output=np.empty(entries)
        self.check(self.api.matrix_overlap_grid(len(dimensions),entries,pointer(matrix),pointer(linear),
            pointer(dimensions),pointer(strides),ground,pointer(output)))
        return output

    def operator_rows(self,states,ids,values,constant,linear):
        states=np.ascontiguousarray(states,dtype=np.int64);ids=np.ascontiguousarray(ids,dtype=np.int64)
        values=np.ascontiguousarray(values,dtype=np.float64)
        constant=np.ascontiguousarray(constant,dtype=np.complex128);linear=np.ascontiguousarray(linear,dtype=np.complex128)
        output=np.empty((len(states),len(constant)),complex)
        self.check(self.api.matrix_operator_rows(len(states),states.shape[1],len(constant),
            *[pointer(a) for a in (states,ids,values,constant,linear,output)]))
        return output

    def contract_orders(self,rows,offsets,indices,coefficients,moments):
        offsets=np.ascontiguousarray(offsets,dtype=np.int64);indices=np.ascontiguousarray(indices,dtype=np.int64)
        coefficients=np.ascontiguousarray(coefficients,dtype=np.complex128)
        moments=np.ascontiguousarray(moments,dtype=np.complex128);output=np.empty((rows,3,moments.shape[1]),complex)
        self.check(self.api.matrix_contract_orders(rows,moments.shape[1],
            *[pointer(a) for a in (offsets,indices,coefficients,moments,output)]))
        return output

    def dipole_ti_batch(self,tree,cache,pending,count,model,gaussian,q,direction,floor,operator):
        pending=np.ascontiguousarray(pending,dtype=np.int64);rows=max(count,len(pending));d=len(cache.linear);n=d//2
        states=np.empty((rows,d),np.int64)
        if len(pending):states[:len(pending)]=pending
        ids=np.empty((rows,d+1),np.int64);amplitudes=np.empty((rows,5));columns=np.empty((rows,4))
        live=np.ascontiguousarray(cache.ht,dtype=np.uint8);totals=np.empty(5);class_weights=np.empty(n+1)
        class_counts=np.empty(n+1,np.int64);counts=np.empty(3,np.uintp);timings=np.empty(3)
        inputs=[np.ascontiguousarray(x,dtype=dtype) for x,dtype in (
            (model.transition_dipole_au,complex),(gaussian['ht_constant'],complex),(gaussian['ht_linear'],complex),
            (q,float),(model.initial_frequencies_cm,float),(model.final_frequencies_cm,float))]
        request=PackedRequest(pointer(states),0,d,pointer(cache.keys),cache.keys.shape[1],len(cache.values),
            pointer(cache.slots),len(cache.slots),pointer(cache.values),cache.count,cache.edge_count,
            pointer(cache.matrix),pointer(cache.linear),pointer(live),cache.bits,int(cache.mask),2,pointer(ids),0,0)
        arguments=(ct.byref(request),tree._native_state.handle,count,len(pending),
            pointer(states),*[pointer(x) for x in inputs],model.origin_cm,direction,floor,
            1 if operator=='FC' else 2 if operator=='HT' else 4,
            *[pointer(x) for x in (amplitudes,columns,totals,class_weights,class_counts,counts,timings)])
        context=self.packed_context(cache)
        if context is None:result=self.api.matrix_dipole_ti_batch(*arguments)
        else:
            with context.lock:result=self.api.matrix_dipole_ti_batch_context(*arguments,context.handle)
        cache.count=request.count;cache.edge_count=request.edge_count
        tree.allocated=int(counts[1]);tree.visited=int(counts[2]);self.check(result)
        kept=int(request.completed);selected=int(counts[0])
        return (states[:kept],states[kept:selected].copy(),amplitudes[:kept],columns[:kept],
            totals,class_weights,class_counts,bool(request.exhausted),timings)

    def prepare_geometry(self,model,temperature_K):
        from .vibronic_native import write_native_harmonic_input
        n=len(model.initial_frequencies_cm);d=2*n
        matrix=np.empty((d,d));linear=np.empty(d);log_ground=np.empty(1);coordinates=np.empty(n*(d+1))
        with tempfile.TemporaryDirectory(prefix='matrix-native-geometry-') as folder:
            path=Path(folder)/'model.dat'
            write_native_harmonic_input(model,path,method='TI',temperature_K=temperature_K,
                maximum_initial_quanta=0,maximum_final_quanta=0,maximum_class=0)
            self.check(self.api.matrix_prepare_gaussian_geometry(str(path).encode(),n,
                *[pointer(x) for x in (matrix,linear,log_ground,coordinates)]))
        return dict(matrix=matrix,linear=linear,log_ground=float(log_ground[0]),coordinates=coordinates)

    def insert_dipole(self,geometry,model):
        n=len(model.initial_frequencies_cm);constant=np.empty(3,complex);linear=np.empty((3,2*n),complex)
        gradient=np.ascontiguousarray(model.transition_gradient_dimensionless_au,dtype=complex)
        momentum=np.ascontiguousarray(model.transition_momentum_dimensionless_au,dtype=complex)
        self.check(self.api.matrix_insert_dipole(n,*[pointer(x) for x in
            (geometry['coordinates'],gradient,momentum,constant,linear)]))
        return dict(matrix=geometry['matrix'],linear=geometry['linear'],log_ground=geometry['log_ground'],
            ht_constant=constant,ht_linear=linear)

    def prepare_bank(self,model,constants,gradients,forms,temperature_K):
        from .vibronic_native import write_native_operator_input
        n=len(model.initial_frequencies_cm);d=2*n;k=len(constants);f=len(forms)
        matrix=np.empty((d,d));linear=np.empty(d);log_ground=np.empty(1);constant=np.empty(k,complex)
        insertion=np.empty((k,d),complex);rules=np.empty(f);norm=np.empty(1);q=np.empty(n)
        with tempfile.TemporaryDirectory(prefix='matrix-native-gaussian-') as folder:
            path=Path(folder)/'model.dat'
            write_native_operator_input(model,path,constants,forms,gradients=gradients,method='TI',
                temperature_K=temperature_K,maximum_initial_quanta=0,maximum_final_quanta=0,
                maximum_class=0,export_profile=False)
            self.check(self.api.matrix_prepare_operator_gaussian(str(path).encode(),n,k,f,pointer(matrix),
                pointer(linear),pointer(log_ground),pointer(constant),pointer(insertion),pointer(rules),pointer(norm),pointer(q)))
        gaussian=dict(matrix=matrix,linear=linear,log_ground=float(log_ground[0]),
            ht_constant=np.zeros(3,complex),ht_linear=np.zeros((3,d),complex))
        return gaussian,constant,insertion,rules,float(norm[0]),q


class NativePackedWorkspace:
    """Cache-owned buffers and transient hash tags, never serialized as state."""
    def __init__(self,library):
        self.library=library;self.lock=threading.RLock()
        self.handle=library.api.matrix_packed_context_create(library.recurrence_mode=='levels',library.recurrence_workers)
        if not self.handle:library.check(-1)

    def close(self):
        with self.lock:
            if self.handle:self.library.api.matrix_packed_context_destroy(self.handle);self.handle=None

    def __del__(self):
        if getattr(self,'handle',None):self.close()


class NativeTreeState:
    """Own the native heap while checkpoint metadata and typed arrays stay shared."""
    def __init__(self,tree,library,heap_arrays):
        self.library=library;costs,nodes,axes=heap_arrays
        names=('costs','values','parents','original','ancestors','axes','ranks')
        self.arrays=tuple(getattr(tree,name) for name in names)
        if any(not array.flags.c_contiguous for array in self.arrays):
            raise ValueError('Contiguous conditional-tree arrays required')
        request=TreeRequest(len(tree.parents),tree.costs.shape[1],tree.budget,len(tree.ancestors),
            *[pointer(a) for a in self.arrays],tree.allocated,tree.visited,pointer(costs),pointer(nodes),pointer(axes),
            len(costs),tree.modes,tree.qi,tree.qf,tree.maximum_class)
        self.handle=library.api.matrix_tree_create(ct.byref(request))
        if not self.handle:library.check(-1)

    def __len__(self):
        return self.library.api.matrix_tree_heap_size(self.handle)

    def native_arrays(self):
        count=len(self);cost=np.empty(count);node=np.empty(count,np.int64);axis=np.empty(count,np.int64)
        self.library.check(self.library.api.matrix_tree_export_heap(self.handle,pointer(cost),pointer(node),pointer(axis)))
        return cost,node,axis

    def batch(self,count):
        output=np.empty((count,len(self.arrays[2])),np.int64);allocated=S();visited=S()
        rows=self.library.api.matrix_tree_batch(self.handle,count,pointer(output),ct.byref(allocated),ct.byref(visited))
        self.library.check(rows);return output[:rows],allocated.value,visited.value

    def __del__(self):
        if getattr(self,'handle',None):
            self.library.api.matrix_tree_destroy(self.handle);self.handle=None


class SavedNativeHeap:
    """Checkpoint-owned heap arrays before the native state takes ownership."""
    def __init__(self,cost,node,axis):
        self.arrays=(cost,node,axis)

    def native_arrays(self):
        return self.arrays

    def __len__(self):
        return len(self.arrays[0])
