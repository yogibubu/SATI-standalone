"""Conservative coercivity certificates for polynomial n-mode contractions."""
from __future__ import annotations

from itertools import product
from fractions import Fraction
import numpy as np


def audit_nmode_polynomial(one_mode_coefficients_cm,one_mode_powers,frequencies_cm,
                           active_pairs=(),contracted_pairs=(),*,active_dimension=0,active_grid=(0.,)):
    """Check the entire supplied polynomial at every path/active-grid point.

    Includes harmonic quadratics and sums repeated operator coefficients.
    Negative/odd leading axis terms prove a nonconfining ray. Remaining mixed
    terms use a weighted Young inequality after normalizing the positive pure
    leading terms. A failed sufficient certificate is reported as unknown,
    never as a proof of instability. No coefficient is clipped or altered.
    """
    raw=frequencies_cm
    if not np.isrealobj(raw):raise ValueError('real positive frequencies required')
    frequency=np.asarray(raw,float)
    if frequency.ndim!=2 or not frequency.size or not np.all(np.isfinite(frequency)) or np.any(frequency<=0):
        raise ValueError('real positive frequencies required')
    paths,modes=frequency.shape
    if len(one_mode_coefficients_cm)!=modes or len(one_mode_powers)!=modes:
        raise ValueError('one polynomial per contracted mode required')
    if isinstance(active_dimension,bool) or not isinstance(active_dimension,(int,np.integer)) or active_dimension<0:
        raise ValueError('nonnegative active dimension required')
    if not np.isrealobj(active_grid):raise ValueError('real active sample grid required')
    grid=np.asarray(active_grid,float)
    if grid.ndim!=1 or not len(grid) or not np.all(np.isfinite(grid)):
        raise ValueError('finite active sample grid required')
    points=np.asarray(list(product(grid,repeat=active_dimension))).reshape(-1,active_dimension) if active_dimension else np.empty((1,0))
    shape=(paths,len(points));terms={}
    def powers(raw,length):
        value=np.asarray(raw)
        if value.size==0:return np.empty((0,length),int)
        if value.ndim!=2 or value.shape[1]!=length or not np.all(np.isfinite(value)) or np.any(value<0) or np.any(value!=np.floor(value)):
            raise ValueError('nonnegative integer polynomial powers required')
        return value.astype(int)
    def coefficients(raw,count):
        if not np.isrealobj(raw):raise ValueError('real finite polynomial coefficients required')
        value=np.asarray(raw,float)
        if value.shape!=(paths,count) or not np.all(np.isfinite(value)):
            raise ValueError('polynomial coefficients must share the finite path grid')
        return value
    def add(word,value):
        key=tuple(word)
        terms[key]=terms.get(key,np.zeros(shape))+np.broadcast_to(value,shape)
        if not np.all(np.isfinite(terms[key])):raise ValueError('nonfinite combined potential coefficients')
    for mode,(raw,words) in enumerate(zip(one_mode_coefficients_cm,one_mode_powers,strict=True)):
        words=powers(words,1);values=coefficients(raw,len(words))
        word=[0]*modes;word[mode]=2;add(word,.5*frequency[:,mode,None])
        for column,power in enumerate(words[:,0]):
            word=[0]*modes;word[mode]=int(power);add(word,values[:,column,None])
    for mode,raw,words,coordinate,position in active_pairs:
        if not 0<=mode<modes or not 0<=coordinate<active_dimension or position not in (0,1):
            raise ValueError('invalid active-pair coordinates')
        words=powers(words,2);values=coefficients(raw,len(words))
        for column,word_pair in enumerate(words):
            word=[0]*modes;word[mode]=int(word_pair[1-position])
            add(word,values[:,column,None]*points[None,:,coordinate]**word_pair[position])
    for first,second,raw,words,position in contracted_pairs:
        if not 0<=first<modes or not 0<=second<modes or first==second or position not in (0,1):
            raise ValueError('invalid contracted-pair coordinates')
        words=powers(words,2);values=coefficients(raw,len(words))
        for column,word_pair in enumerate(words):
            word=[0]*modes;word[first]=int(word_pair[position]);word[second]=int(word_pair[1-position])
            add(word,values[:,column,None])
    failures=[];minimum_margin=1.
    for path in range(paths):
        for point in range(len(points)):
            degrees=[];leading=[];axis_bad=[]
            for mode in range(modes):
                pure=[(word[mode],value[path,point]) for word,value in terms.items()
                      if word[mode]>0 and sum(word)==word[mode] and value[path,point]!=0]
                degree,coefficient=max(pure,default=(0,0.))
                degrees.append(degree);leading.append(coefficient)
                if degree<2 or degree%2 or coefficient<=0:axis_bad.append(mode)
            if axis_bad:
                failures.append(dict(path_index=path,active_point_index=point,status='nonconfining_axis',modes=axis_bad))
                continue
            margin=np.ones(modes);unknown=False
            for word,values in terms.items():
                coefficient=values[path,point]
                if coefficient==0 or sum(x>0 for x in word)<2:continue
                if coefficient>0 and all(x%2==0 for x in word):continue
                weights=np.asarray(word)/np.asarray(degrees)
                degree=sum(Fraction(int(a),int(b)) for a,b in zip(word,degrees,strict=True))
                if degree<1:continue
                if degree>1:
                    unknown=True;continue
                relative=abs(coefficient)*np.exp(-float(weights@np.log(leading)))
                margin-=relative*weights
            minimum_margin=min(minimum_margin,float(min(margin)))
            if unknown or np.any(margin<=np.finfo(float).eps*max(16,4*len(terms))):
                failures.append(dict(path_index=path,active_point_index=point,status='mixed_coercivity_not_certified'))
    return dict(schema='matrix.gf.nmode_polynomial_stability.v1',certified=not failures,
        path_points=paths,active_points=len(points),checked_points=paths*len(points),
        minimum_normalized_margin=minimum_margin,failures=failures,
        certificate='positive pure leading terms and normalized weighted Young inequality',
        scope='sufficient global confinement certificate in contracted coordinates at supplied active/path points; not interpolation or spectral convergence')


__all__=['audit_nmode_polynomial']
