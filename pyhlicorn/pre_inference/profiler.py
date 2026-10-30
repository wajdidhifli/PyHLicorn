"""Diagnostic candidate-space profiler for PyHLicorn.

Evaluates co-regulator candidate itemset counts across parameter grids (discretization
sigma and min_coreg_support) using the streaming size-bounded vertical miner.
Counts candidates without materializing candidate-output collections in memory.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from .preprocessing import PreProcess
from .vertical_mining import (
    bitsets_from_matrix,
    bounded_closed_sets,
    minimum_count,
    row_bitsets,
)


def _cardinality_label(k: int) -> str:
    """Human-readable column label for cardinality k."""
    names = {1: "singletons", 2: "pairs", 3: "triples", 4: "quadruples", 5: "quintuples"}
    return names.get(k, f"size_{k}")


def _profile_for_discrete_matrix(
    disc_mat: pd.DataFrame,
    sig_val: Optional[float],
    tf_set: Set[str],
    n_samples: int,
    supports: Sequence[float],
    min_gene_support: float,
    max_coreg: int,
    timeout_per_cell: Optional[float],
    max_candidates_per_cell: Optional[int],
    card_keys: List[str],
) -> List[Dict[str, Any]]:
    """Helper scoped to a single discretization matrix.
    
    All TF arrays, transaction matrices, packed bitsets, and row masks allocated
    inside this function go out of scope and are released upon return.
    """
    records: List[Dict[str, Any]] = []

    # Sparsity across full matrix
    total_cells = disc_mat.size
    nonzero_cells = int(np.count_nonzero(disc_mat.to_numpy(copy=False)))
    sparsity_pct = (nonzero_cells / total_cells * 100.0) if total_cells > 0 else 0.0

    # TF eligibility filter (identical to hlicorn.py)
    common_tfs = [t for t in disc_mat.index if t in tf_set]
    if not common_tfs:
        for supp in supports:
            rec: Dict[str, Any] = {
                "sigma": sig_val if sig_val is not None else "supplied",
                "min_coreg_support": supp,
                "eligible_tfs": 0,
                "active_cell_pct": round(sparsity_pct, 2),
            }
            for ck in card_keys:
                rec[ck] = 0
            rec.update({
                "total_candidates": 0,
                "within_group_multitf": False if max_coreg > 1 else False,
                "status": "no eligible TFs in matrix",
                "enumeration_seconds": 0.0,
            })
            records.append(rec)
        return records

    tf_disc = disc_mat.loc[common_tfs]
    tf_active = np.count_nonzero(tf_disc.to_numpy(copy=False), axis=1)
    eligible_mask = tf_active > (n_samples * min_gene_support)
    eligible_tfs = [t for t, ok in zip(common_tfs, eligible_mask) if ok]

    if len(eligible_tfs) < 2:
        for supp in supports:
            rec = {
                "sigma": sig_val if sig_val is not None else "supplied",
                "min_coreg_support": supp,
                "eligible_tfs": len(eligible_tfs),
                "active_cell_pct": round(sparsity_pct, 2),
            }
            for ck in card_keys:
                rec[ck] = len(eligible_tfs) if ck == "singletons" else 0
            rec.update({
                "total_candidates": len(eligible_tfs),
                "within_group_multitf": False if max_coreg > 1 else False,
                "status": "< 2 eligible TFs passed support",
                "enumeration_seconds": 0.0,
            })
            records.append(rec)
        return records

    # Build signed transaction matrix (2*n_samples rows, |eligible_tfs| cols)
    reg_disc_exp = disc_mat.loc[eligible_tfs]
    reg_values = reg_disc_exp.to_numpy(copy=False)
    n_tfs = reg_values.shape[0]
    bit_values = np.empty((2 * n_samples, n_tfs), dtype=bool)
    np.equal(reg_values.T, 1, out=bit_values[:n_samples])
    np.equal(reg_values.T, -1, out=bit_values[n_samples:])

    tx_count = bit_values.shape[0]
    singleton_min = minimum_count(min_gene_support, tx_count)
    columns = bitsets_from_matrix(bit_values)

    # Profile across requested supports
    for supp in supports:
        coreg_min = minimum_count(supp, tx_count)
        t_cell_start = time.perf_counter()
        deadline = (t_cell_start + timeout_per_cell) if timeout_per_cell is not None else None

        def make_check(dl):
            if dl is None:
                return lambda: None
            def _check():
                if time.perf_counter() > dl:
                    raise TimeoutError("Per-support candidate enumeration timeout reached")
            return _check

        check_fn = make_check(deadline)
        counts = {k: 0 for k in range(1, max_coreg + 1)}
        total = 0
        status = "complete"

        try:
            # 1. Enforce candidate limits during singletons
            for col in columns:
                check_fn()
                if col and col.bit_count() >= singleton_min:
                    counts[1] += 1
                    total += 1
                    if max_candidates_per_cell is not None and total >= max_candidates_per_cell:
                        status = f"incomplete (>= {max_candidates_per_cell:,} candidate cap)"
                        break

            # 2. Multi-TF size-bounded closed sets (only if cap not reached and max_coreg > 1)
            if status == "complete" and max_coreg > 1:
                item_ids = [i for i, col in enumerate(columns) if col and col.bit_count() >= coreg_min]
                # row_bitsets inside try block so timeout cannot escape
                rows = row_bitsets(bit_values, item_ids, check_fn) if len(item_ids) >= 2 else None
                for group, _ in bounded_closed_sets(
                    columns, tx_count, coreg_min, 2, max_coreg, check=check_fn, row_masks=rows
                ):
                    card = len(group)
                    if card in counts:
                        counts[card] += 1
                    total += 1
                    if max_candidates_per_cell is not None and total >= max_candidates_per_cell:
                        status = f"incomplete (>= {max_candidates_per_cell:,} candidate cap)"
                        break
        except TimeoutError:
            status = f"incomplete (timeout > {timeout_per_cell:.1f}s)"

        elapsed = time.perf_counter() - t_cell_start

        multi_tf_count = sum(counts.get(k, 0) for k in range(2, max_coreg + 1))
        if multi_tf_count > 0:
            has_multitf: Optional[bool] = True
        elif status == "complete":
            has_multitf = False
        else:
            has_multitf = None  # Unknown because enumeration was cut short

        rec = {
            "sigma": sig_val if sig_val is not None else "supplied",
            "min_coreg_support": supp,
            "eligible_tfs": len(eligible_tfs),
            "active_cell_pct": round(sparsity_pct, 2),
        }
        for k in range(1, max_coreg + 1):
            rec[_cardinality_label(k)] = counts.get(k, 0)

        rec.update({
            "total_candidates": total,
            "within_group_multitf": has_multitf,
            "status": status,
            "enumeration_seconds": round(elapsed, 3),
        })
        records.append(rec)

    return records


def profile_candidates(
    expression: pd.DataFrame,
    tfs: Union[Sequence[str], pd.Series, pd.DataFrame],
    discrete: Optional[pd.DataFrame] = None,
    sigmas: Optional[Sequence[float]] = None,
    supports: Optional[Sequence[float]] = None,
    min_gene_support: float = 0.1,
    max_coreg: int = 3,
    timeout_per_cell: Optional[float] = 10.0,
    max_candidates_per_cell: Optional[int] = 5_000_000,
) -> pd.DataFrame:
    """Profile candidate co-regulator counts across parameter combinations.

    Reuses exact PyHLicorn input normalization, TF filtering, signed transaction
    construction, and size-bounded mining semantics. No materialized candidate-output
    collection is stored in memory; counts are accumulated directly in registers.

    Parameters
    ----------
    expression : pd.DataFrame
        Continuous gene expression matrix (genes as rows, samples as columns).
    tfs : sequence, Series, or DataFrame
        Transcription factor identifiers.
    discrete : pd.DataFrame, optional
        Pre-discretized expression matrix (-1, 0, 1). If supplied, sigmas are
        ignored and profiling is conducted strictly on this matrix.
    sigmas : sequence of float, optional
        Standard deviation thresholds for discretization when discrete is None.
        Must be finite positive numbers. Defaults to (1.0, 1.25, 1.5).
    supports : sequence of float, optional
        Candidate co-regulator support thresholds to evaluate.
        Must be finite numbers in [0.0, 1.0]. Defaults to (0.1, 0.2, 0.3).
    min_gene_support : float, default 0.1
        Minimum sample activity fraction required for a TF to be eligible (in [0.0, 1.0]).
    max_coreg : int, default 3
        Maximum co-regulator group size (integer >= 1).
    timeout_per_cell : float or None, default 10.0
        Wall-clock cutoff in seconds for per-support candidate enumeration.
        Does not cover initial matrix discretization or TF preprocessing.
        Must be a finite positive number or None.
    max_candidates_per_cell : int or None, default 5,000,000
        Maximum candidate enumeration cutoff for any single grid point.
        Must be a positive integer or None.

    Returns
    -------
    pd.DataFrame
        Table summarizing eligible TFs, sparsity, counts by cardinality,
        multi-TF candidate availability, enumeration status, and runtime.
    """
    # 1. Parameter validation
    if not isinstance(max_coreg, int) or isinstance(max_coreg, bool) or max_coreg < 1:
        raise ValueError("max_coreg must be an integer >= 1.")
    
    if not isinstance(min_gene_support, (int, float)) or not math.isfinite(min_gene_support) or not (0.0 <= min_gene_support <= 1.0):
        raise ValueError(f"min_gene_support must be a finite float in [0.0, 1.0], got {min_gene_support}")

    if timeout_per_cell is not None:
        if not isinstance(timeout_per_cell, (int, float)) or not math.isfinite(timeout_per_cell) or timeout_per_cell <= 0:
            raise ValueError(f"timeout_per_cell must be a finite positive number or None, got {timeout_per_cell}")
        timeout_per_cell = float(timeout_per_cell)

    if max_candidates_per_cell is not None:
        if not isinstance(max_candidates_per_cell, int) or isinstance(max_candidates_per_cell, bool) or max_candidates_per_cell < 1:
            raise ValueError(f"max_candidates_per_cell must be a positive integer or None, got {max_candidates_per_cell}")

    if supports is None:
        supports = (0.1, 0.2, 0.3)
    supports_list = []
    for s in supports:
        if not isinstance(s, (int, float)) or not math.isfinite(s) or not (0.0 <= s <= 1.0):
            raise ValueError(f"Support values must be finite numbers in [0.0, 1.0], got {s}")
        supports_list.append(float(s))
    supports_list = sorted(supports_list)

    # 2. Expression validation (identical to hlicorn.py)
    if not isinstance(expression, pd.DataFrame):
        raise TypeError("expression must be a pandas DataFrame.")
    if expression.empty:
        raise ValueError("Numerical expression must be nonempty and finite.")
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("Expression gene and sample identifiers must be unique.")
    if not all(isinstance(index, str) for index in expression.index):
        raise ValueError("All indices of numerical_expression must be of type str.")
    
    # Check finiteness in blocks
    block_rows = max(1, (32 * 1024 * 1024) // max(1, 8 * expression.shape[1]))
    for start in range(0, len(expression), block_rows):
        if not np.isfinite(expression.iloc[start:start + block_rows].to_numpy(dtype=float)).all():
            raise ValueError("Numerical expression must be nonempty and finite.")

    n_samples = expression.shape[1]
    if n_samples < 2:
        raise ValueError("Expression matrix must contain at least 2 samples.")

    # 3. Normalize TF list (exact inference normalization)
    if isinstance(tfs, pd.DataFrame):
        if tfs.shape[1] > 0:
            tf_list = tfs.iloc[:, 0].dropna().tolist()
        else:
            tf_list = tfs.index.dropna().tolist()
    elif isinstance(tfs, (pd.Series, pd.Index)):
        tf_list = tfs.dropna().tolist()
    elif isinstance(tfs, (list, tuple, set)):
        tf_list = list(tfs)
    else:
        raise TypeError("tfs must be a list, tuple, set, Series, or DataFrame.")
    tf_set = set(str(tf) for tf in tf_list)

    # 4. Discrete validation (if supplied)
    if discrete is not None:
        if not isinstance(discrete, pd.DataFrame):
            raise TypeError("discrete must be a pandas DataFrame.")
        if (not expression.index.equals(discrete.index)
                or not expression.columns.equals(discrete.columns)):
            raise ValueError("Numerical and discrete expression must have identical ordered gene and sample labels.")
        for start in range(0, len(discrete), block_rows):
            if not np.isin(discrete.iloc[start:start + block_rows].to_numpy(), [-1, 0, 1]).all():
                raise ValueError("Discrete expression must contain only -1, 0, and 1, with no missing values.")
        discrete = discrete.astype(np.int8, copy=False)
        sigmas_to_process: List[Optional[float]] = [None]
    else:
        if sigmas is None:
            sigmas = (1.0, 1.25, 1.5)
        sigmas_list = []
        for sig in sigmas:
            if not isinstance(sig, (int, float)) or not math.isfinite(sig) or sig <= 0:
                raise ValueError(f"Discretization sigma must be a finite positive number, got {sig}")
            sigmas_list.append(float(sig))
        sigmas_to_process = sorted(sigmas_list)

    card_keys = [_cardinality_label(k) for k in range(1, max_coreg + 1)]
    records: List[Dict[str, Any]] = []

    # 5. Process sigmas sequentially with helper scoping (cleans up arrays before next sigma)
    for sig_val in sigmas_to_process:
        if sig_val is not None:
            disc_mat = PreProcess.discretize_gene_expression(expression, threshold=sig_val)
        else:
            disc_mat = discrete

        cell_records = _profile_for_discrete_matrix(
            disc_mat=disc_mat,
            sig_val=sig_val,
            tf_set=tf_set,
            n_samples=n_samples,
            supports=supports_list,
            min_gene_support=min_gene_support,
            max_coreg=max_coreg,
            timeout_per_cell=timeout_per_cell,
            max_candidates_per_cell=max_candidates_per_cell,
            card_keys=card_keys,
        )
        records.extend(cell_records)
        del disc_mat

    return pd.DataFrame.from_records(records)


def format_profile_table(df: pd.DataFrame) -> str:
    """Format candidate profile results as an aligned, neutral diagnostic report."""
    if df.empty:
        return "No profiling records available."

    # Identify cardinality columns present in df
    known_cards = []
    col_set = set(df.columns)
    k = 1
    while True:
        cl = _cardinality_label(k)
        if cl in col_set:
            known_cards.append((k, cl))
            k += 1
        else:
            break

    # Build header dynamically
    header_parts = [
        f"{'Sigma':<10}",
        f"{'Support':<9}",
        f"{'Eligible TFs':<13}",
        f"{'Active %':<10}",
    ]
    for k, cl in known_cards:
        header_parts.append(f"{cl.capitalize():<12}")
    header_parts.append(f"{'Total':<14}")
    header_parts.append(f"{'Enum (s)':<10}")
    header_parts.append(f"{'Multi-TF?':<11}")
    header_line = " ".join(header_parts)
    divider = "-" * len(header_line)

    lines = [divider, header_line, divider]

    for r in df.itertuples(index=False):
        sig_str = f"{r.sigma:.2f}" if isinstance(r.sigma, (int, float)) else str(r.sigma)
        supp_str = f"{r.min_coreg_support:.2f}"
        
        status_val = str(getattr(r, "status", "complete"))
        total_val = getattr(r, "total_candidates", 0)
        is_incomplete = status_val.startswith("incomplete")
        
        # Only format as lower bound >= N* if status is incomplete
        if is_incomplete:
            total_str = f">={total_val:,}*"
        else:
            total_str = f"{total_val:,}"
        
        multitf_val = getattr(r, "within_group_multitf", None)
        if multitf_val is True:
            multi_str = "Yes"
        elif multitf_val is False:
            multi_str = "None (<=1)"
        else:
            multi_str = "Unknown*"

        row_parts = [
            f"{sig_str:<10}",
            f"{supp_str:<9}",
            f"{getattr(r, 'eligible_tfs', 0):<13}",
            f"{getattr(r, 'active_cell_pct', 0.0):<10.1f}",
        ]
        for k, cl in known_cards:
            val = getattr(r, cl, 0)
            val_str = f"{val:,}"
            row_parts.append(f"{val_str:<12}")
        row_parts.append(f"{total_str:<14}")
        sec_val = getattr(r, "enumeration_seconds", 0.0)
        row_parts.append(f"{sec_val:<10.3f}")
        row_parts.append(f"{multi_str:<11}")
        lines.append(" ".join(row_parts))

    lines.append(divider)
    lines.append("Notes:")
    lines.append("  • Active %: percentage of non-zero entries (+1 / -1) in the ternary discrete expression matrix.")
    lines.append("  • Cardinality columns: count of candidate activator / repressor complexes mined in Phase 1.")
    lines.append("  • Multi-TF?: indicates whether candidate co-activators/co-repressors with >= 2 TFs exist at that support.")
    lines.append("    When 'None (<=1)', Phase 1 yields only singletons; models can still pair 1 activator and 1 repressor.")
    lines.append("  • Asterisk (*): enumeration was interrupted by timeout or candidate cap; count is a lower bound, and")
    lines.append("    absence of multi-TF candidates is 'Unknown'.")
    lines.append("  • Timing: 'Enum (s)' measures per-support candidate enumeration time; it excludes initial matrix discretization.")
    lines.append("  • Memory note: profiling accumulates counts directly without materializing candidate collections in memory.")
    lines.append("  • Downstream regression runtime depends on target-specific concordance screening and sample size.")
    lines.append(divider)
    return "\n".join(lines)
