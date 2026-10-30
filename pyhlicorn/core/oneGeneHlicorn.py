"""Exact target screening, native pair scoring, and fitting of ranked models."""
import numpy as np
import pandas as pd
from ..during_inference import __eand, Linear_Regression
from ..during_inference.__comblicorn import select_disjoint
from .._serialization import normalize_frame


def get_regs(reg_arr, indices, ncarr, search_thresh):
    true_rows = np.count_nonzero(np.bitwise_and.reduce(reg_arr[:, indices], axis=1))
    if true_rows / ncarr >= search_thresh:
        return indices


def _empty_result(linear_model):
    columns = ['Target', 'coact', 'corep']
    if linear_model is not None:
        columns += ['Coef_Acts', 'Coef_Reps', 'Coef_coActs', 'Coef_coReps', 'R2', 'RMSE']
    return pd.DataFrame(columns=columns)


def oneGeneHlicorn(gene, gene_disc_arr, reg_disc_exp, items, trans_reg_bit_data,
                  search_thresh, reg_num_exp, Y, nGRN, linear_model, convert,
                  cv_folds=None, max_candidate_pairs=None, diagnostics=None):
    if convert:
        gene_disc_arr = np.asarray(gene_disc_arr)
        reg_disc_exp = np.asarray(reg_disc_exp)
        trans_reg_bit_data = np.asarray(trans_reg_bit_data)
        reg_num_exp = np.asarray(reg_num_exp)
        Y = np.asarray(Y)
    shift = len(gene_disc_arr)
    positive = np.flatnonzero(gene_disc_arr == 1)
    negative = np.flatnonzero(gene_disc_arr == -1)
    ncarr = len(positive) + len(negative)
    if ncarr == 0:
        return _empty_result(linear_model)
    act_rows = trans_reg_bit_data[np.concatenate([positive, negative + shift])]
    rep_rows = trans_reg_bit_data[np.concatenate([positive + shift, negative])]
    # A conjunction cannot have greater support than any one of its members.
    # Divide as in get_regs to preserve its floating-point threshold semantics.
    act_singletons = act_rows.sum(axis=0) / ncarr >= search_thresh
    rep_singletons = rep_rows.sum(axis=0) / ncarr >= search_thresh
    if hasattr(items, 'iter_indices'):
        candidates = items.iter_indices()
        names = items.names
        def resolve(indices):
            return [names[int(i)] for i in indices]
    else:
        # Compatibility with low-level callers supplying (indices, names) pairs.
        name_map = {}
        def index_iterator():
            for indices, names in items:
                name_map.update(zip(indices, names))
                yield indices
        candidates = index_iterator()
        def resolve(indices):
            return [name_map[int(i)] for i in indices]
    act_groups, rep_groups, act_exp, rep_exp = [], [], [], []
    total = act_checked = rep_checked = multi = multi_checks = 0
    for indices in candidates:
        total += 1
        is_multi = len(indices) > 1
        multi += is_multi
        if np.all(act_singletons[indices]):
            act_checked += 1
            multi_checks += is_multi
            if get_regs(act_rows, indices, ncarr, search_thresh) is not None:
                act_groups.append(np.array(indices, copy=True))
                act_exp.append(np.asarray(__eand(indices, reg_disc_exp), dtype=np.int8).ravel())
        if np.all(rep_singletons[indices]):
            rep_checked += 1
            multi_checks += is_multi
            if get_regs(rep_rows, indices, ncarr, search_thresh) is not None:
                rep_groups.append(np.array(indices, copy=True))
                rep_exp.append(np.asarray(__eand(indices, reg_disc_exp), dtype=np.int8).ravel())
    pairs = (len(act_groups) + 1) * (len(rep_groups) + 1)
    if max_candidate_pairs is not None and pairs > max_candidate_pairs:
        raise MemoryError(f'Target {gene}: {pairs:,} candidate pairs exceeds '
                          f'max_candidate_pairs={max_candidate_pairs:,}. '
                          'No candidates were silently discarded.')
    # Empty rows allow activation-only and repression-only models.
    act_groups.append(())
    rep_groups.append(())
    neutral = np.zeros(shift, dtype=np.int32)
    act_exp.append(neutral)
    rep_exp.append(neutral)
    act_values = np.vstack(act_exp, dtype=np.int32).ravel()
    rep_values = np.vstack(rep_exp, dtype=np.int32).ravel()
    del act_exp, rep_exp
    rep_values[rep_values == 1] = 2
    iact, irep = select_disjoint(act_values, act_groups, rep_values, rep_groups,
                                gene_disc_arr, nGRN, diagnostics=diagnostics)
    if diagnostics is not None:
        diagnostics.update(candidates=total, multi_candidates=multi,
                           act_conjunction_checks=act_checked, rep_conjunction_checks=rep_checked,
                           multi_conjunction_checks=multi_checks,
                           multi_conjunction_checks_without_pruning=2*multi,
                           pair_capacity=pairs)
    gact = [act_groups[int(i)] for i in iact]
    grep = [rep_groups[int(i)] for i in irep]
    # No TF-name lists for rejected candidates or unselected models.
    act_names = [resolve(indices) for indices in gact]
    rep_names = [resolve(indices) for indices in grep]
    if linear_model is None:
        return pd.DataFrame(dict(Target=[gene]*len(iact), coact=act_names, corep=rep_names))
    coefficients = np.full((len(iact), 6), pd.NA, dtype=object)
    coefficients = Linear_Regression(gact, grep, coefficients, reg_num_exp, Y,
                                     linear_model, cv_folds=cv_folds)
    result = pd.DataFrame.from_records(coefficients, columns=[
        'Coef_Acts', 'Coef_Reps', 'Coef_coActs', 'Coef_coReps', 'R2', 'RMSE'])
    result.insert(0, 'Target', gene)
    result.insert(1, 'coact', act_names)
    result.insert(2, 'corep', rep_names)
    if not result.empty:
        result = result.sort_values(['R2', 'RMSE'], ascending=[False, True])
    return normalize_frame(result)
