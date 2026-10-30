import numpy as np
import pandas as pd

from ..cpylibs import comblicorn

def __comblicorn(co_act_exp, co_act, co_rep_exp, co_rep, gene_disc_exp):

    coactexp = co_act_exp.astype(np.int32)
    ncoacts = len(co_act)

    corepexp = co_rep_exp.astype(np.int32)
    ncoreps = len(co_rep)

    gexp = gene_disc_exp.astype(dtype=np.int32)
    nsamples = len(gexp)

    # Vector to store MAE results
    result = np.full((len(co_act) * len(co_rep)), -1.0, dtype=np.float64)

    # Vectors to store indices of coactivator and corepressor
    iact = np.full((len(co_act) * len(co_rep)), -1, dtype=np.int32)
    irep = np.full((len(co_act) * len(co_rep)), -1, dtype=np.int32)
    
    result, iact, irep = comblicorn.combLicorn(coactexp, ncoacts, corepexp, ncoreps, gexp, nsamples, result, iact, irep)

    return result, iact, irep


def _membership(groups):
    """Sort for intersection tests only; feature ordering is left unchanged."""
    offsets = np.empty(len(groups) + 1, dtype=np.int64)
    offsets[0] = 0
    for i, group in enumerate(groups):
        offsets[i + 1] = offsets[i] + len(group)
    flat = np.empty(int(offsets[-1]), dtype=np.int32)
    for i, group in enumerate(groups):
        flat[offsets[i]:offsets[i+1]] = np.sort(group)
    return flat, offsets


def score_disjoint(co_act_exp, act_groups, co_rep_exp, rep_groups, gene_disc_exp):
    nact, nrep = len(act_groups), len(rep_groups)
    pairs = nact * nrep
    if pairs > np.iinfo(np.int32).max:
        raise ValueError('Too many candidate pairs for the native kernel')
    result = np.empty(pairs, dtype=np.float64)
    iact = np.empty(pairs, dtype=np.int32)
    irep = np.empty(pairs, dtype=np.int32)
    af, ao = _membership(act_groups)
    rf, ro = _membership(rep_groups)
    result, iact, irep, count = comblicorn.combLicornDisjoint(
        np.ascontiguousarray(co_act_exp, dtype=np.int32), nact,
        np.ascontiguousarray(co_rep_exp, dtype=np.int32), nrep,
        np.ascontiguousarray(gene_disc_exp, dtype=np.int32), len(gene_disc_exp),
        af, ao, rf, ro, result, iact, irep)
    return result[:count], iact[:count] - 1, irep[:count] - 1


def select_disjoint(co_act_exp, act_groups, co_rep_exp, rep_groups,
                    gene_disc_exp, nGRN, block_pairs=262144, diagnostics=None):
    """Exact min-rank selection with bounded scoring buffers and all boundary ties.

    Two passes reuse the validated native kernel. The first accumulates an
    integer total-error histogram; the second retrieves selected indices in
    original row-major order. Retained indices and final models are output-sized,
    not bounded by block_pairs. No full Cartesian score/rank array is allocated.
    """
    if not isinstance(block_pairs, int) or isinstance(block_pairs, bool) or block_pairs < 1:
        raise ValueError('block_pairs must be a positive integer')
    gene = np.ascontiguousarray(gene_disc_exp, dtype=np.int32)
    samples = len(gene)
    if samples < 1:
        raise ValueError('At least one sample is required')
    act = np.ascontiguousarray(co_act_exp, dtype=np.int32).reshape(len(act_groups), samples)
    rep = np.ascontiguousarray(co_rep_exp, dtype=np.int32).reshape(len(rep_groups), samples)
    nact, nrep = len(act_groups), len(rep_groups)

    def blocks():
        # When splitting repressor columns, visit one activator at a time to
        # preserve precisely the original kernel's row-major output order.
        rows = max(1, block_pairs // max(1, nrep))
        for i in range(0, nact, rows):
            end = min(nact, i + rows)
            for j in range(0, nrep, block_pairs):
                stop = min(nrep, j + block_pairs)
                scores, ia, ir = score_disjoint(act[i:end].ravel(), act_groups[i:end],
                    rep[j:stop].ravel(), rep_groups[j:stop], gene)
                # Native scores are integer error/nsamples, with error<=2*nsamples.
                errors = np.rint(scores * samples).astype(np.int64)
                yield errors, ia, ir, i, j

    histogram = np.zeros(2*samples+1, dtype=np.int64)
    small = None
    for errors, ai, ri, i, j in blocks():
        histogram += np.bincount(errors, minlength=len(histogram))
        if nact*nrep <= block_pairs:
            small = (errors, ai, ri, i, j)
    valid = int(histogram.sum())
    occupied = np.flatnonzero(histogram)
    cutoff = -1
    if len(occupied):
        # min-rank(s) = 1 + count(scores strictly smaller than s).
        prior = np.cumsum(histogram) - histogram
        eligible = occupied[(prior[occupied] + 1) <= (1 if nGRN is None else nGRN)]
        if len(eligible):
            cutoff = int(eligible[-1])
    selected_a, selected_r = [], []
    if cutoff >= 0:
        for errors, ia, ir, i, j in ([small] if small is not None else blocks()):
            keep = errors <= cutoff
            if np.any(keep):
                selected_a.append(ia[keep] + i)
                selected_r.append(ir[keep] + j)
    ia = np.concatenate(selected_a) if selected_a else np.empty(0, dtype=np.int32)
    ir = np.concatenate(selected_r) if selected_r else np.empty(0, dtype=np.int32)
    if diagnostics is not None:
        diagnostics.update(scored_valid_pairs=valid, selected_pairs=len(ia),
                           pair_scoring_block_limit=block_pairs,
                           selection_error_cutoff=cutoff,
                           pair_scoring_passes=2 if cutoff >= 0 and small is None else 1)
    return ia, ir
