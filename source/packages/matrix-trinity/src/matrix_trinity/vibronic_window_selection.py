"""Distinct per-window occupation rankings with one exact-key union of lines."""
import numpy as np
from numba import njit
from .vibronic_selection import ConditionalOccupationTree,_packed_find


@njit(cache=True,nogil=True)
def _admit(states,keys,slots,count,bits):
    indices=np.empty(len(states),np.int64);kept=0
    for row in range(len(states)):
        key=np.zeros(keys.shape[1],np.uint64)
        for k in range(states.shape[1]):
            key[k*bits//64]|=np.uint64(states[row,k])<<np.uint64(k*bits%64)
        found,slot=_packed_find(key,keys,slots)
        if found<0:
            keys[count]=key;slots[slot]=count;count+=1;indices[kept]=row;kept+=1
    return indices[:kept],count


def ranking_bias(target,model,gaussian,temperature,direction,requested):
    forest=target is not None and target.window_rankings and len(target.windows)>1
    if requested is None:
        if forest:return np.array([target.automatic_bias(model,gaussian,temperature,direction,i) for i in range(len(target.windows))])
        return target.automatic_bias(model,gaussian,temperature,direction) if target is not None else 0.
    values=np.asarray(requested,float)
    if forest:
        try:values=np.broadcast_to(values,(len(target.windows),)).copy()
        except ValueError as error:raise ValueError('one ranking bias per window required') from error
    elif values.ndim:raise ValueError('a scalar ranking bias is required for a single tree')
    if not np.all(np.isfinite(values)):raise ValueError('finite ranking bias required')
    return values if forest else float(values)


class WindowOccupationForest:
    """Round-robin service of unfinished windows, shared recurrence outside.

    Deduplication compares the entire packed occupation vector after hashing.
    A dependency already present in the integral cache is NOT a selected line.
    """
    is_window_forest=True

    def __init__(self,model,temperature,maximum_per_mode,budget,*,biases,target,fraction,packing_cap,**controls):
        self.children=[ConditionalOccupationTree(model,temperature,maximum_per_mode,budget,
            energy_bias_per_cm=float(bias),**controls) for bias in biases]
        self.budget=budget;self.visited=0;self.cursor=0;self.last_window=0;self.duplicates=0
        self.modes=len(model.initial_frequencies_cm);self.qi=controls['qi'];self.qf=controls['qf'];self.maximum_class=controls['maximum_class']
        self.bits=next(b for b in (1,2,4,8,16,32) if b>=int(packing_cap).bit_length())
        self.keys=np.empty((budget,(2*self.modes*self.bits+63)//64),np.uint64)
        self.slots=np.full(1<<int(np.ceil(np.log2(max(2,budget/.65)))),-1,np.int64)
        self.attach(target,fraction)

    def attach(self,target,fraction):self.target=target;self.fraction=fraction

    @property
    def heap(self):return self

    @property
    def allocated(self):return sum(tree.allocated for tree in self.children)

    def __len__(self):return sum(len(tree.heap) for tree in self.children if tree.visited<tree.budget)

    def enable_native(self,library):
        for tree in self.children:tree.enable_native(library)

    def batch(self,count):
        while len(self) and self.visited<self.budget:
            live=[i for i,tree in enumerate(self.children) if tree.heap and tree.visited<tree.budget]
            unfinished=~self.target.window_ready(self.fraction)
            choices=[i for i in live if unfinished[i]] or live
            i=min(choices,key=lambda k:(k-self.cursor)%len(self.children))
            self.cursor=(i+1)%len(self.children);self.last_window=i
            states=self.children[i].batch(min(count,self.budget-self.visited))
            ids,total=_admit(states,self.keys,self.slots,self.visited,self.bits)
            self.duplicates+=len(states)-len(ids);self.visited=total
            if len(ids):return states[ids]
        return np.empty((0,2*self.modes),np.int64)

    def statistics(self):
        return dict(rankings=len(self.children),unique_candidates=self.visited,duplicate_candidates_removed=self.duplicates,
            per_window_proposal_visits=[tree.visited for tree in self.children],
            per_window_proposal_budget_exhausted=[bool(tree.visited>=tree.budget and tree.heap) for tree in self.children],
            shared_integral_cache=True,
            collision_checked_full_occupation_keys=True)

    def checkpoint_arrays(self):
        from .vibronic_checkpoint import _heap_arrays
        arrays={'tree_forest_seen':self.keys[:self.visited]};states=[]
        for i,tree in enumerate(self.children):
            prefix=f'tree_forest_{i}_'
            arrays.update({prefix+name:getattr(tree,name) for name in ('costs','values','parents','original')})
            arrays.update({prefix+name:getattr(tree,name)[:tree.allocated+1] for name in ('ancestors','axes','ranks')})
            for name,value in zip(('heap_cost','heap_node','heap_axis'),_heap_arrays(tree.heap)):arrays[prefix+name]=value
            states.append({name:getattr(tree,name) for name in ('allocated','visited','modes','qi','qf','maximum_class')})
        state=dict(type='window_forest_v1',children=states,visited=self.visited,bits=self.bits,cursor=self.cursor,
            last_window=self.last_window,duplicates=self.duplicates)
        return arrays,state

    @classmethod
    def restore(cls,state,arrays,budget,native_library):
        from .vibronic_checkpoint import _rehash,_restore_heap
        from .vibronic_native_kernels import SavedNativeHeap
        self=cls.__new__(cls);self.budget=budget;self.children=[]
        for name in ('visited','bits','cursor','last_window','duplicates'):setattr(self,name,state[name])
        if any(child['visited']>budget for child in state['children']):raise ValueError('resume budget is below a saved window ranking')
        for i,saved in enumerate(state['children']):
            tree=ConditionalOccupationTree.__new__(ConditionalOccupationTree);tree.budget=budget
            for name,value in saved.items():setattr(tree,name,value)
            prefix=f'tree_forest_{i}_'
            for name in ('costs','values','parents','original'):setattr(tree,name,arrays[prefix+name])
            for name in ('ancestors','axes','ranks'):
                value=np.zeros(2*budget+2,np.int64);old=arrays[prefix+name];value[:len(old)]=old;setattr(tree,name,value)
            heap=[arrays[prefix+name] for name in ('heap_cost','heap_node','heap_axis')]
            tree.heap=SavedNativeHeap(*heap) if native_library is not None else _restore_heap(*heap)
            self.children.append(tree)
        for name in ('modes','qi','qf','maximum_class'):setattr(self,name,getattr(self.children[0],name))
        old=arrays['tree_forest_seen'];self.keys=np.empty((budget,old.shape[1]),np.uint64);self.keys[:len(old)]=old
        self.slots=np.full(1<<int(np.ceil(np.log2(max(2,budget/.65)))),-1,np.int64)
        _rehash(self.keys,self.slots,len(old));return self
