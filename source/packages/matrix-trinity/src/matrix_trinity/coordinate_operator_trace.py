"""Shared real coordinate-operator cross traces on finite active eigenstates."""
import numpy as np
from numba import njit, prange
from .vibronic_harmonic import PHASE_PER_CM_FS


@njit(parallel=True,cache=True)
def _cross_histograms(amplitudes,populations,bin_ids,pairs,bin_count):
    out=np.zeros((len(pairs),bin_count))
    for pair in prange(len(pairs)):
        a,b=pairs[pair]
        for f in range(amplitudes.shape[1]):
            for i in range(amplitudes.shape[2]):
                out[pair,bin_ids[f,i]]+=amplitudes[a,f,i]*amplitudes[b,f,i]*populations[i]
    return out


def symmetric_operator_cross_traces(amplitudes,initial_energies,final_energies,
                                    populations,times_fs,*,energy_bin_cm=1.,pair_batch=32):
    """Return upper-triangle cross traces; caller doubles off-diagonal terms.

    Retains signed operator interference. No positive weights are thresholded.
    Pair batching bounds histogram and time-transform workspace; each energy
    histogram is contracted once for all supplied times.
    """
    v=np.asarray(amplitudes,float);ei=np.asarray(initial_energies,float);ef=np.asarray(final_energies,float)
    p=np.asarray(populations,float);times=np.asarray(times_fs,float)
    if (v.ndim!=3 or v.shape[1:]!=(len(ef),len(ei)) or p.shape!=ei.shape
            or energy_bin_cm<=0 or pair_batch<1 or np.any(p<0)
            or not all(np.all(np.isfinite(x)) for x in (v,ei,ef,p,times))):
        raise ValueError('Matching finite real operators, energies, populations and times required')
    energy=ei[None,:]-ef[:,None];low=np.floor(energy.min()/energy_bin_cm)*energy_bin_cm
    ids=np.floor((energy-low)/energy_bin_cm).astype(np.int64)
    bins=int(ids.max()+1);center=low+(np.arange(bins)+.5)*energy_bin_cm
    pairs=np.array([(a,b) for a in range(len(v)) for b in range(a,len(v))],np.int64)
    traces=np.empty((len(pairs),len(times)),complex)
    phase=np.exp(-1j*PHASE_PER_CM_FS*center[:,None]*times)
    for start in range(0,len(pairs),pair_batch):
        stop=min(start+pair_batch,len(pairs))
        hist=_cross_histograms(v,p,ids,pairs[start:stop],bins)
        traces[start:stop]=hist@phase
    return pairs,traces
