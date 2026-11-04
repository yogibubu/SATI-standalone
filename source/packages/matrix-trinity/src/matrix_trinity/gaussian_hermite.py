"""Shared complex Hermite recurrences extracted from the Raman TD backend.

The graph and polynomial insertion algebra are unchanged.
"""
import numpy as np
from numba import njit


def hermite_coefficients(matrix, linear, constant, plan):
    """Evaluate the zero-property case through the existing Raman recurrence."""
    rows, edge_offsets, edges, offsets, terms = plan
    if np.any(np.diff(offsets) != 1):
        raise ValueError('zero-insertion Hermite plan required')
    values = _hermite_recurrence_values(matrix, linear, constant, rows, edge_offsets, edges)
    return values[terms[:, 0].astype(np.int64)]


def hermite_recurrence_plan(states,degree,maximum_integrals):
    """Coordinate-independent recurrence graph plus polynomial insertions."""
    d=states.shape[1];known={(0,)*d:0};rows=[(0,0,0.)];edges=[[]]
    def ensure(state):
        stack=[(state,False)]
        while stack:
            current,ready=stack.pop()
            if current in known:continue
            j=next(k for k,v in enumerate(current) if v);previous=list(current);previous[j]-=1
            dependencies=[(tuple(previous),-1,1/np.sqrt(current[j]))]
            for k,value in enumerate(previous):
                if value:
                    previous[k]-=1;dependencies.append((tuple(previous),k,np.sqrt(value/current[j])));previous[k]+=1
            if not ready:
                stack.append((current,True))
                stack.extend((parent,False) for parent,_,_ in reversed(dependencies) if parent not in known)
            else:
                if len(known)>=maximum_integrals:raise MemoryError('Raman TD external Hermite cache budget exceeded')
                known[current]=len(known);rows.append((j,known[dependencies[0][0]],dependencies[0][2]))
                edges.append([(known[parent],k,factor) for parent,k,factor in dependencies[1:]])
        return known[state]
    offsets=[0];terms=[]
    for state in states.tolist():
        terms.append((ensure(tuple(state)),-1,-1,1.))
        if degree:
            for k,v in enumerate(state):
                if not v:continue
                state[k]-=1;terms.append((ensure(tuple(state)),k,-1,np.sqrt(v)))
                for l,w in enumerate(state):
                    if w:
                        state[l]-=1;terms.append((ensure(tuple(state)),k,l,np.sqrt(v*w)));state[l]+=1
                state[k]+=1
        offsets.append(len(terms))
    edge_offsets=[0];flat=[]
    for group in edges:flat.extend(group);edge_offsets.append(len(flat))
    return (np.asarray(rows,float),np.asarray(edge_offsets,np.int64),np.asarray(flat,float).reshape(-1,3),
        np.asarray(offsets,np.int64),np.asarray(terms,float))


@njit(cache=True,nogil=True)
def _hermite_recurrence_values(matrix,r,constant,plan_rows,edge_offsets,edges):
    values=np.empty(len(plan_rows),np.complex128);values[0]=constant
    for index in range(1,len(values)):
        j=int(plan_rows[index,0]);parent=int(plan_rows[index,1])
        value=r[j]*values[parent]*plan_rows[index,2]
        for edge in range(edge_offsets[index],edge_offsets[index+1]):
            value+=matrix[j,int(edges[edge,1])]*values[int(edges[edge,0])]*edges[edge,2]
        values[index]=value
    return values


@njit(cache=True,nogil=True)
def hermite_polynomial_coefficients(matrix,r,constant,left,right,covariance,plan_rows,edge_offsets,edges,offsets,terms):
    values=_hermite_recurrence_values(matrix,r,constant,plan_rows,edge_offsets,edges)
    result=np.zeros((len(offsets)-1,3,3),np.complex128)
    for row in range(len(result)):
        for term in range(offsets[row],offsets[row+1]):
            k=int(terms[term,1]);l=int(terms[term,2]);value=values[int(terms[term,0])]*terms[term,3]
            for a in range(3):
                for b in range(3):
                    if k<0:polynomial=left[a,0]*right[b,0]+covariance[a,b]
                    elif l<0:polynomial=left[a,k+1]*right[b,0]+left[a,0]*right[b,k+1]
                    else:polynomial=left[a,k+1]*right[b,l+1]
                    result[row,a,b]+=value*polynomial
    return result
