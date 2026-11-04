"""Deterministic best-first occupation selection with exact-strength stopping.

The conditional Gaussian proposal orders visits; it is not a bound on coupled intensities.
No branch is discarded because of a weak parent. Completeness is certified
only by the exact positive strength sum rule after full FC/HT evaluation.
"""
import heapq
import numpy as np
from numba import njit, types
from numba.typed import List
from .vibronic_harmonic import harmonic_thermal_factors, harmonic_gaussian_generator


@njit(cache=True, nogil=True)
def _gaussian_number_marginal(vx, vp, displacement, maximum):
    """Diagonal Fock coefficients of a one-mode reduced Gaussian density."""
    determinant = (vx+.5)*(vp+.5)
    squeeze = (vx-vp)/(2*determinant)
    thermal = max(0., (vx*vp-.25)/determinant)
    source = displacement/(np.sqrt(2)*(vx+.5))
    table = np.zeros((maximum+1, maximum+1))
    table[0, 0] = np.exp(-displacement*displacement/(2*(vx+.5)))/np.sqrt(determinant)
    for j in range(1, maximum+1):
        table[0, j] = (source*table[0, j-1]+(squeeze*np.sqrt(j-1)*table[0, j-2] if j > 1 else 0.))/np.sqrt(j)
    for i in range(1, maximum+1):
        for j in range(maximum+1):
            value = source*table[i-1, j]
            if i > 1: value += squeeze*np.sqrt(i-1)*table[i-2, j]
            if j: value += thermal*np.sqrt(j)*table[i-1, j-1]
            table[i, j] = value/np.sqrt(i)
    # This is a proposal, not the evaluated spectrum. Remove only negative
    # roundoff from its density coefficients; physical strengths are unchanged.
    return np.maximum(np.diag(table), 0.)


@njit(cache=True, nogil=True)
def _pair_density_coefficients(matrix, source, vacuum, first_max, second_max):
    dims = np.array([first_max+1, second_max+1, first_max+1, second_max+1])
    strides = np.array([dims[1]*dims[2]*dims[3], dims[2]*dims[3], dims[3], 1])
    table = np.zeros(np.prod(dims)); table[0] = vacuum
    for index in range(1, len(table)):
        counts = np.empty(4, np.int64); rem = index
        for k in range(4): counts[k] = rem//strides[k]; rem %= strides[k]
        j = 0
        while counts[j] == 0: j += 1
        previous = index-strides[j]; counts[j] -= 1
        value = source[j]*table[previous]
        for k in range(4):
            if counts[k]: value += matrix[j, k]*np.sqrt(counts[k])*table[previous-strides[k]]
        table[index] = value/np.sqrt(counts[j]+1)
    result = np.empty((first_max+1, second_max+1))
    for i in range(first_max+1):
        for j in range(second_max+1): result[i, j] = max(0., table[i*(strides[0]+strides[2])+j*(strides[1]+strides[3])])
    return result


@njit(cache=True, nogil=True)
def _pure_pair_coefficients(matrix, source, vacuum, first_max, second_max):
    table = np.zeros((first_max+1, second_max+1)); table[0, 0] = np.sqrt(vacuum)
    for i in range(first_max+1):
        for j in range(second_max+1):
            if i:
                value = source[0]*table[i-1, j]
                if i > 1: value += matrix[0, 0]*np.sqrt(i-1)*table[i-2, j]
                if j: value += matrix[0, 1]*np.sqrt(j)*table[i-1, j-1]
                table[i, j] = value/np.sqrt(i)
            elif j:
                value = source[1]*table[0, j-1]
                if j > 1: value += matrix[1, 1]*np.sqrt(j-1)*table[0, j-2]
                table[0, j] = value/np.sqrt(j)
    return table*table


def gaussian_pair_marginal(vx, vp, mean, first_max, second_max):
    dx = vx+.5*np.eye(2); dp = vp+.5*np.eye(2)
    ix = np.linalg.inv(dx); ip = np.linalg.inv(dp)
    vacuum = float(np.exp(-.5*mean @ ix @ mean)/np.sqrt(np.linalg.det(dx)*np.linalg.det(dp)))
    if np.max(abs(vx @ vp-.25*np.eye(2))) < 1e-12:
        # Reduced pure pairs (notably thermal initial/final twins) require
        # only a two-dimensional amplitude table, not a four-dimensional one.
        matrix = np.eye(2)-ix
        source = ix @ mean/np.sqrt(2)
        return _pure_pair_coefficients(matrix, source, vacuum, first_max, second_max)
    if (first_max+1)**2*(second_max+1)**2 > 100000:
        # Cauchy extraction of the positive photon-number generating function.
        # Per-axis radial damping bounds the alias contribution by 1e-12;
        # eightfold padding limits roundoff amplification in retained orders.
        # These are proposal coefficients only, never physical intensities.
        from scipy.fft import next_fast_len, ifft2
        caps = (first_max, second_max)
        lengths = [next_fast_len(8*(cap+1)) if cap else 1 for cap in caps]
        logs = [np.log(1e-12)/length if cap else 0. for cap, length in zip(caps, lengths, strict=True)]
        circles = [np.exp(log-2j*np.pi*np.arange(length)/length) if cap else np.zeros(1, complex)
                   for cap, length, log in zip(caps, lengths, logs, strict=True)]
        z = np.stack(np.broadcast_arrays(circles[0][:, None], circles[1][None, :]), axis=-1).reshape(-1, 2)
        signal = np.empty(len(z), complex)
        identity = np.eye(2)
        px = identity-2*np.linalg.inv(identity+2*vx)
        pp = identity-2*np.linalg.inv(identity+2*vp)
        logzero = np.linalg.slogdet(identity+2*vx)[1]+np.linalg.slogdet(identity+2*vp)[1]
        for start in range(0, len(z), 4096):
            zz = z[start:start+4096]
            ex = np.linalg.eigvals(zz[:, :, None]*px)
            ep = np.linalg.eigvals(zz[:, :, None]*pp)
            logdet = logzero+np.log1p(-ex).sum(axis=1)+np.log1p(-ep).sum(axis=1)
            sx = (1+zz)[:, :, None]*identity+2*(1-zz)[:, :, None]*vx
            solved = np.linalg.solve(sx, ((1-zz)*mean)[:, :, None])[:, :, 0]
            signal[start:start+len(zz)] = np.exp(2*np.log(2)-.5*logdet-solved @ mean)
        coefficients = ifft2(signal.reshape(lengths)).real[:first_max+1, :second_max+1]
        coefficients *= np.exp(-np.arange(first_max+1)[:, None]*logs[0]-np.arange(second_max+1)[None, :]*logs[1])
        return np.maximum(coefficients, 0.)
    a = .5*(ip-ix); b = np.eye(2)-.5*(ip+ix)
    matrix = np.block([[a, b], [b, a]])
    source = np.tile(ix @ mean/np.sqrt(2), 2)
    return _pair_density_coefficients(matrix, source, vacuum, first_max, second_max)


def conditional_occupation_proposal(model, temperature_K, maximum_per_mode, *, operator_constant=None, operator_linear=None,
        proposal='gaussian_c2',gaussian=None,native_library=None,
        energy_bias_per_cm=0.,channel_weights=None):
    """Maximum-correlation tree of exact Gaussian C2 number marginals.

    Max-product messages give conditional subtree MAP completions. The resulting
    distribution is a ranking proposal only, not a physical factorization.
    """
    if proposal not in ('gaussian_c2','operator_c2'):
        raise ValueError('proposal must be gaussian_c2 or operator_c2')
    q=harmonic_thermal_factors(model.initial_frequencies_cm,temperature_K)
    g=harmonic_gaussian_generator(model) if gaussian is None else gaussian;n=len(q);d=2*n
    if not np.isfinite(energy_bias_per_cm):
        raise ValueError('finite selection energy bias required')
    thermal=np.r_[np.sqrt(q),np.ones(n)]
    if energy_bias_per_cm:
        # Exponential tilting changes ranking only: proposal probabilities are
        # multiplied by exp(-bias * (E_final-E_initial)). Exact physical
        # amplitudes, populations and omission certificates use the source g.
        exponent=-.5*energy_bias_per_cm*np.r_[-model.initial_frequencies_cm,model.final_frequencies_cm]
        if np.max(abs(exponent))>300:
            raise ValueError('selection energy bias exceeds safe exponential range')
        thermal*=np.exp(exponent)
    r=g['linear']*thermal
    matrix=thermal[:,None]*g['matrix']*thermal[None,:];eye=np.eye(d)
    if energy_bias_per_cm and np.max(abs(np.linalg.eigvalsh(matrix)))>=1-1e-12:
        raise ValueError('energy-biased Gaussian proposal is not normalizable')
    vx=.5*np.linalg.solve(eye-matrix,eye+matrix);vx=.5*(vx+vx.T)
    vp=.5*np.linalg.solve(eye+matrix,eye-matrix);vp=.5*(vp+vp.T)
    mean=np.sqrt(2)*np.linalg.solve(eye-matrix,r)
    covariance=.5*(vx*vx+vp*vp)+mean[:,None]*vx*mean[None,:]
    covariance[np.diag_indices(d)]-=.25
    std=np.sqrt(np.maximum(0.,np.diag(covariance)));denom=std[:,None]*std[None,:]
    weights=np.divide(abs(covariance),denom,out=np.zeros_like(covariance),where=denom>0)
    guide_linear=g['ht_linear'] if operator_linear is None else operator_linear
    guide_constant=model.transition_dipole_au+g['ht_constant'] if operator_constant is None else operator_constant
    if channel_weights is not None:
        weights_array=np.asarray(channel_weights,float)
        if (proposal!='operator_c2' or weights_array.shape!=guide_constant.shape
                or not np.all(np.isfinite(weights_array)) or np.any(weights_array<=0)):
            raise ValueError('positive finite per-channel weights require operator_c2')
        scale=np.sqrt(weights_array/weights_array.max())
        guide_constant=guide_constant*scale;guide_linear=guide_linear*scale[:,None]
    ht=np.sum(abs(guide_linear*thermal[None,:])**2,axis=0)
    norm=float(np.sum(abs(guide_constant)**2)+ht.sum())
    operator_blocks=proposal=='operator_c2' and norm>0
    if operator_blocks:
        from .gaussian_operator_marginals import (gaussian_operator_number_marginal,
            gaussian_operator_number_moments)
        scaled=guide_linear*thermal[None,:]
        active=np.flatnonzero(np.any(matrix!=0,axis=1)|(r!=0)|np.any(scaled!=0,axis=0))
        reduced_matrix=matrix[np.ix_(active,active)];reduced_linear=r[active]
        reduced_h=scaled[:,active]/np.sqrt(norm);reduced_c=guide_constant/np.sqrt(norm)
        reduced_log=g['log_ground']+.5*np.log1p(-q).sum()
        lookup={int(mode):i for i,mode in enumerate(active)}
        if len(active):
            moments=gaussian_operator_number_moments(reduced_matrix,reduced_linear,reduced_log,reduced_c,reduced_h)
            covariance=np.zeros((d,d));covariance[np.ix_(active,active)]=moments['covariance']
            std=np.sqrt(np.maximum(0.,np.diag(covariance)));denom=std[:,None]*std[None,:]
            weights=np.divide(abs(covariance),denom,out=np.zeros_like(covariance),where=denom>0)
        def block(indices,caps):
            result=gaussian_operator_number_marginal(reduced_matrix,reduced_linear,reduced_log,
                reduced_c,reduced_h,np.array([lookup[i] for i in indices]),np.asarray(caps,dtype=int))
            return result['strengths']
    marginals=[];caps=[]
    for k in range(d):
        if operator_blocks and k in lookup:
            p=block([k],[maximum_per_mode])
            p=np.maximum(p,np.finfo(float).tiny)
        elif operator_blocks:
            p=np.zeros(maximum_per_mode+1);p[0]=1.
        else:
            p=(_gaussian_number_marginal(vx[k,k],vp[k,k],mean[k],maximum_per_mode) if native_library is None
                else native_library.marginal1(vx[k,k],vp[k,k],mean[k],maximum_per_mode))
        cap=int(np.searchsorted(np.cumsum(p),(1-1e-12)*p.sum()))
        caps.append(min(maximum_per_mode,cap+(1 if ht[k]>0 and not operator_blocks else 0)));marginals.append(p)
    # Prim's tree, rooted at the mode with greatest photon-number variance.
    original=[int(np.argmax(std))];parents=[-1];remaining=set(range(d))-set(original)
    best=weights[original[0]].copy();owner=np.zeros(d,np.int64)
    while remaining:
        child=min(remaining,key=lambda k:(-best[k],k));remaining.remove(child)
        parents.append(int(owner[child]));original.append(child)
        for k in remaining:
            if weights[child,k]>best[k]:best[k]=weights[child,k];owner[k]=len(original)-1
    length=maximum_per_mode+1
    logs=np.full((d,length,length),-np.inf)
    for axis,i in enumerate(original):
        if axis==0:
            p=marginals[i][:caps[i]+1].copy()
            if ht[i]>0 and not operator_blocks:p[1:]+=(ht[i]/norm)*np.arange(1,len(p))*p[:-1].copy()
            p/=p.sum();logs[0,0,:len(p)]=np.log(p,where=p>0,out=np.full_like(p,-np.inf))
        else:
            j=original[parents[axis]];idx=np.array([j,i])
            if operator_blocks:
                if j not in lookup or i not in lookup:
                    p=np.outer(marginals[j][:caps[j]+1],marginals[i][:caps[i]+1])
                else:
                    p=block([j,i],[caps[j],caps[i]])
                p=np.maximum(p,np.finfo(float).tiny)
            else:
                p=(gaussian_pair_marginal(vx[np.ix_(idx,idx)],vp[np.ix_(idx,idx)],mean[idx],caps[j],caps[i])
                    if native_library is None else native_library.marginal2(vx[np.ix_(idx,idx)],vp[np.ix_(idx,idx)],mean[idx],caps[j],caps[i]))
            guide=p.copy()
            if ht[j]>0 and not operator_blocks:guide[1:]+=(ht[j]/norm)*np.arange(1,len(guide))[:,None]*p[:-1]
            if ht[i]>0 and not operator_blocks:guide[:,1:]+=(ht[i]/norm)*np.arange(1,guide.shape[1])[None,:]*p[:,:-1]
            normalization=guide.sum(axis=1)
            guide=np.divide(guide,normalization[:,None],out=np.zeros_like(guide),where=normalization[:,None]>0)
            logs[axis,:guide.shape[0],:guide.shape[1]]=np.log(guide,where=guide>0,out=np.full_like(guide,-np.inf))
    messages=np.zeros((d,length));utility=np.zeros((d,length))
    costs=np.full_like(logs,np.inf);values=np.zeros(logs.shape,np.int64)
    for axis in range(d-1,-1,-1):
        scores=logs[axis]+utility[axis][None,:]
        ranks=np.argsort(-scores,axis=1,kind='stable')
        ordered=np.take_along_axis(scores,ranks,axis=1);maximum=ordered[:,0]
        valid=np.isfinite(maximum)
        costs[axis,valid]=maximum[valid,None]-ordered[valid]
        values[axis]=ranks;messages[axis]=maximum
        if axis:utility[parents[axis]]+=maximum
    if not np.isfinite(messages[0,0]):raise FloatingPointError('conditional Gaussian proposal has no finite root')
    return costs,values,np.asarray(parents,np.int64),np.asarray(original,np.int64)


@njit(cache=True,nogil=True)
def _conditional_values(node,ancestors,axes,ranks,parents,values):
    assigned=np.zeros(len(parents),np.bool_);selection=np.zeros(len(parents),np.int64)
    while node:
        axis=axes[node]
        if not assigned[axis]:selection[axis]=ranks[node];assigned[axis]=True
        node=ancestors[node]
    physical=np.zeros(len(parents),np.int64)
    for axis in range(len(parents)):
        parent_value=0 if parents[axis]<0 else physical[parents[axis]]
        physical[axis]=values[axis,parent_value,selection[axis]]
    return selection,physical


@njit(cache=True,nogil=True)
def _conditional_batch(heap,ancestors,axes,ranks,costs,values,parents,original,
        allocated,visited,budget,batch_size,modes,qi,qf,maximum_class):
    result=np.empty((batch_size,2*modes),np.int64);kept=0
    while heap and visited<budget and kept<batch_size:
        cost,node,append_axis=heapq.heappop(heap)
        selection,physical=_conditional_values(node,ancestors,axes,ranks,parents,values)
        if append_axis>=0:
            # This virtual iterator emits the next cheapest append axis for
            # one fixed prefix. Only one iterator per parent is queued.
            base=node;last=-1 if base==0 else axes[base]
            previous_parent=0 if parents[append_axis]<0 else physical[parents[append_axis]]
            previous_penalty=costs[append_axis,previous_parent,1]
            next_axis=-1;next_penalty=np.inf
            for j in range(last+1,len(parents)):
                parent_value=0 if parents[j]<0 else physical[parents[j]]
                penalty=costs[j,parent_value,1]
                if (penalty>previous_penalty or (penalty==previous_penalty and j>append_axis)):
                    if penalty<next_penalty or (penalty==next_penalty and (next_axis<0 or j<next_axis)):
                        next_axis=j;next_penalty=penalty
            if next_axis>=0 and np.isfinite(next_penalty):
                heapq.heappush(heap,(cost-previous_penalty+next_penalty,base,next_axis))
            allocated+=1;node=allocated;ancestors[node]=base;axes[node]=append_axis;ranks[node]=1
            selection,physical=_conditional_values(node,ancestors,axes,ranks,parents,values)
        visited+=1
        state=np.zeros(2*modes,np.int64)
        for j in range(len(parents)):state[original[j]]=physical[j]
        if (np.sum(state[:modes])<=qi and np.sum(state[modes:])<=qf
            and np.count_nonzero(state[:modes])<=maximum_class
            and np.count_nonzero(state[modes:])<=maximum_class):
            result[kept]=state;kept+=1
        last=-1 if node==0 else axes[node]
        if last>=0:
            rank=ranks[node];parent_value=0 if parents[last]<0 else physical[parents[last]]
            if rank+1<costs.shape[2] and np.isfinite(costs[last,parent_value,rank+1]):
                allocated+=1;ancestors[allocated]=node;axes[allocated]=last;ranks[allocated]=rank+1
                heapq.heappush(heap,(cost+costs[last,parent_value,rank+1]-costs[last,parent_value,rank],allocated,-1))
        next_axis=-1;next_penalty=np.inf
        for j in range(last+1,len(parents)):
            parent_value=0 if parents[j]<0 else physical[parents[j]]
            penalty=costs[j,parent_value,1]
            if penalty<next_penalty:next_axis=j;next_penalty=penalty
        if next_axis>=0 and np.isfinite(next_penalty):heapq.heappush(heap,(cost+next_penalty,node,next_axis))
    return result[:kept],allocated,visited


class ConditionalOccupationTree:
    """Exact k-best traversal of a conditional-tree proposal with lazy appends.

    Max-product completion makes all deviations nonnegative. Disjoint prefix
    regions ensure uniqueness; no amplitude or weak-parent pruning is used.
    """
    def __init__(self,model,temperature_K,maximum_per_mode,budget,*,qi,qf,maximum_class, operator_constant=None, operator_linear=None,
            proposal='gaussian_c2',gaussian=None,native_library=None,
            energy_bias_per_cm=0.,channel_weights=None):
        self.costs,self.values,self.parents,self.original=conditional_occupation_proposal(model,temperature_K,maximum_per_mode,
            operator_constant=operator_constant,operator_linear=operator_linear,proposal=proposal,gaussian=gaussian,
            native_library=native_library,energy_bias_per_cm=energy_bias_per_cm,channel_weights=channel_weights)
        self.ancestors=np.zeros(2*budget+2,np.int64);self.axes=np.zeros(2*budget+2,np.int64);self.ranks=np.zeros(2*budget+2,np.int64)
        self.heap=None
        if native_library is None:
            self.heap=List.empty_list(types.Tuple((types.float64,types.int64,types.int64)))
            self.heap.append((0.,0,-1))
        self.allocated=0;self.visited=0;self.budget=budget
        self.modes=len(model.initial_frequencies_cm);self.qi=qi;self.qf=qf;self.maximum_class=maximum_class
        if native_library is not None:self.enable_native(native_library)

    def enable_native(self,library):
        from .vibronic_checkpoint import _heap_arrays
        from .vibronic_native_kernels import NativeTreeState
        arrays=(np.zeros(1),np.zeros(1,np.int64),np.full(1,-1,np.int64)) if self.heap is None else _heap_arrays(self.heap)
        state=NativeTreeState(self,library,arrays)
        self.native_library=library;self._native_state=state;self.heap=state

    def batch(self,count):
        if getattr(self,'native_library',None) is not None:
            result,self.allocated,self.visited=self._native_state.batch(count)
            return result
        result,self.allocated,self.visited=_conditional_batch(self.heap,self.ancestors,self.axes,self.ranks,
            self.costs,self.values,self.parents,self.original,self.allocated,self.visited,self.budget,count,
            self.modes,self.qi,self.qf,self.maximum_class)
        return result


@njit(cache=True,nogil=True)
def _packed_find(key,keys,slots):
    # Hashing accelerates lookup only. Every occupied slot is checked against
    # the complete packed occupation vector, so collisions cannot alias states.
    hash_value=np.uint64(1469598103934665603)
    for word in key:
        mixed=word+np.uint64(0x9e3779b97f4a7c15)
        mixed=(mixed^(mixed>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        mixed=(mixed^(mixed>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        mixed ^= mixed>>np.uint64(31)
        hash_value=(hash_value^mixed)*np.uint64(1099511628211)

    slot=np.int64(hash_value & np.uint64(len(slots)-1))
    while slots[slot]>=0:
        node=slots[slot];same=True
        for k in range(len(key)):
            if key[k]!=keys[node,k]:same=False;break
        if same:return node,slot
        slot=(slot+1)&(len(slots)-1)
    return -1,slot


@njit(cache=True,nogil=True)
def _packed_ensure(key,keys,slots,values,count,edge_count,matrix,linear,bits,mask,stack,
        cursor,axis,denominator,sums,pending):
    known,_=_packed_find(key,keys,slots)
    if known>=0:return known,count,edge_count,False
    depth=0;stack[0]=key;cursor[0]=-2
    while depth>=0:
        if cursor[depth]==-2:
            minimum=np.int64(mask)+1;j=0
            for k in range(len(linear)):
                value=np.int64((stack[depth,(k*bits)//64]>>np.uint64((k*bits)%64)) & mask)
                if value and value<=minimum:minimum=value;j=k
            axis[depth]=j;denominator[depth]=np.sqrt(minimum);sums[depth]=0.;cursor[depth]=-1
        j=axis[depth]
        if cursor[depth]>len(linear)-1:
            if count>=len(values):return -1,count,edge_count,True
            node=count;count+=1;keys[node]=stack[depth];values[node]=sums[depth]
            _,slot=_packed_find(stack[depth],keys,slots);slots[slot]=node
            depth-=1
            if depth<0:return node,count,edge_count,False
            sums[depth]+=pending[depth]*values[node]
            continue
        k=cursor[depth];cursor[depth]+=1
        parent=stack[depth].copy();parent[(j*bits)//64]-=np.uint64(1)<<np.uint64((j*bits)%64)
        if k==-1:
            factor=linear[j]/denominator[depth]
            if factor==0.:continue
        else:
            occupation=np.int64((parent[(k*bits)//64]>>np.uint64((k*bits)%64)) & mask)
            if not occupation or matrix[j,k]==0.:continue
            factor=matrix[j,k]*np.sqrt(occupation)/denominator[depth]
            parent[(k*bits)//64]-=np.uint64(1)<<np.uint64((k*bits)%64)
        edge_count+=1
        known,_=_packed_find(parent,keys,slots)
        if known>=0:sums[depth]+=factor*values[known]
        else:
            if depth+1>=len(stack):raise RuntimeError('packed recurrence depth exceeds occupation bounds')
            pending[depth]=factor;depth+=1;stack[depth]=parent;cursor[depth]=-2
    return -1,count,edge_count,True


@njit(cache=True,nogil=True)
def _packed_batch(states,keys,slots,values,count,edge_count,matrix,linear,ht,bits,mask,maximum_depth):
    ids=np.full((len(states),states.shape[1]+1),-1,np.int64)
    stack=np.empty((maximum_depth,len(keys[0])),np.uint64)
    cursor=np.empty(maximum_depth,np.int64);axis=np.empty(maximum_depth,np.int64)
    denominator=np.empty(maximum_depth);sums=np.empty(maximum_depth);pending=np.empty(maximum_depth)
    completed=0
    for row in range(len(states)):
        key=np.zeros(len(keys[0]),np.uint64)
        for k in range(states.shape[1]):key[(k*bits)//64] |= np.uint64(states[row,k])<<np.uint64((k*bits)%64)
        node,count,edge_count,exhausted=_packed_ensure(key,keys,slots,values,count,edge_count,matrix,linear,bits,mask,stack,cursor,axis,denominator,sums,pending)
        if exhausted:return ids[:completed],count,edge_count,True
        ids[row,0]=node
        for k in range(states.shape[1]):
            if not states[row,k] or not ht[k]:continue
            parent=key.copy();parent[(k*bits)//64]-=np.uint64(1)<<np.uint64((k*bits)%64)
            node,count,edge_count,exhausted=_packed_ensure(parent,keys,slots,values,count,edge_count,matrix,linear,bits,mask,stack,cursor,axis,denominator,sums,pending)
            if exhausted:return ids[:completed],count,edge_count,True
            ids[row,k+1]=node
        completed+=1
    return ids,count,edge_count,False


class PackedHermiteCache:
    """Compiled exact-key cache and iterative recurrence with bounded storage.

    All key words are compared after hashing. Dependency edges are contracted
    immediately and discarded; their amplitudes remain shared in the cache.
    This avoids both Python dependency construction and persistent edge arrays.
    """
    def __init__(self,gaussian,linear,maximum_integrals,maximum_quanta,*,native_library=None):
        self.matrix=gaussian['matrix'];self.linear=gaussian['linear'];self.ht=np.any(linear!=0,axis=0)
        self.bits=next(b for b in (1,2,4,8,16,32,64) if b>=int(maximum_quanta).bit_length())
        if self.bits==64:raise ValueError('occupation packing requires quanta below 2**32')
        self.mask=np.uint64((1<<self.bits)-1)
        words=(len(self.linear)*self.bits+63)//64
        self.keys=np.empty((maximum_integrals,words),np.uint64);self.keys[0]=0
        capacity=1<<int(np.ceil(np.log2(max(2,maximum_integrals/.65))))
        self.slots=np.full(capacity,-1,np.int64)
        self.values=np.empty(maximum_integrals);self.values[0]=np.exp(gaussian['log_ground'])
        self.count=1;self.edge_count=0;self.maximum_quanta=maximum_quanta
        self.native_library=native_library
        if native_library is not None:
            self.matrix=np.ascontiguousarray(self.matrix,dtype=np.float64)
            self.linear=np.ascontiguousarray(self.linear,dtype=np.float64)
            native_library.rehash(self)
        else:
            _,slot=_packed_find(self.keys[0],self.keys,self.slots);self.slots[slot]=0

    def evaluate(self,states):
        states=np.asarray(states)
        if states.ndim!=2 or states.shape[1]!=len(self.linear) or not np.issubdtype(states.dtype,np.integer):
            raise ValueError("integer occupation rows of the Gaussian dimension required")
        if np.any(states<0) or np.any(states>self.maximum_quanta):
            raise ValueError("occupations outside the declared exact key bounds")
        if self.native_library is not None:return self.native_library.packed_batch(self,states)
        ids,self.count,self.edge_count,exhausted=_packed_batch(states,self.keys,self.slots,self.values,
            self.count,self.edge_count,self.matrix,self.linear,self.ht,self.bits,self.mask,
            int(states.sum(axis=1).max())+2 if len(states) else 2)
        return ids,exhausted
