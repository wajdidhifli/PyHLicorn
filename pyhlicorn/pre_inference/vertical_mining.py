"""Streaming size-bounded co-regulator mining over vertical transaction bitsets.

Inference retains closed sets below max_coreg and all frequent sets at that cap.
The global closed_sets helper below remains available for postprocessing.

Close-by-One style DFS: prefix canonicality avoids a global duplicate repository.
Closedness is checked against ALL frequent attributes, including extensions past
zmax. A closure larger than zmax prunes its entire descendant branch. This is a
memory-first reference implementation, not an implementation of every LCM
optimization. Enumeration remains potentially very expensive.
"""
import math
import sys
import time
import numpy as np
from .candidate_store import AdaptiveCandidateWriter


def minimum_count(proportion, transactions):
    # Signed transactions have 2*n rows: support p/2 requires ceil(p*n) occurrences.
    # A small tolerance prevents a floating point integer-boundary overshoot.
    return max(1, math.ceil((proportion*50) * transactions / 100 - 1e-12))


def bitsets_from_matrix(matrix, check=lambda: None):
    columns = []
    for i in range(matrix.shape[1]):
        check()
        packed = np.packbits(matrix[:, i], bitorder='little')
        columns.append(int.from_bytes(packed.tobytes(), 'little'))
    return columns


def row_bitsets(matrix, item_ids, check=lambda: None):
    """Packed transaction attributes; build with a bounded conversion block."""
    result = []
    block_rows = max(1, (512*1024)//max(1, len(item_ids)))
    for start in range(0, len(matrix), block_rows):
        check()
        packed = np.packbits(matrix[start:start+block_rows, item_ids], axis=1, bitorder='little')
        result.extend(int.from_bytes(row.tobytes(), 'little') for row in packed)
    return result


def closed_sets(columns, transactions, minimum, zmin, zmax, check=lambda: None, stats=None, maximal=False, row_masks=None):
    # Zero-occurrence items are not in PyFIM's observed item universe.
    active = [(i, tid) for i, tid in enumerate(columns) if tid and tid.bit_count() >= minimum]
    item_ids = [i for i,_ in active]
    tids = [tid for _,tid in active]
    count = len(tids)
    if not count or transactions < minimum:
        return
    all_rows = (1 << transactions)-1

    def closure(extent, parent, pivot):
        if row_masks is not None:
            # Intersect transaction attribute masks. Stop as soon as the known
            # generator is reached: its attributes cannot be lost from this extent.
            base = parent if pivot is None else parent | (1 << pivot)
            intent = (1 << count)-1
            rows = extent
            iterations = 0
            while rows:
                bit = rows & -rows
                intent &= row_masks[bit.bit_length()-1]
                if intent == base:
                    break
                rows ^= bit
                iterations += 1
                if iterations % 32 == 0:
                    check()
            if intent.bit_count() > zmax:
                return None
            if pivot is not None and (intent & ~parent & ((1 << pivot)-1)):
                return None
            return intent
        intent = parent
        size = parent.bit_count()
        for i, tid in enumerate(tids):
            if i % 64 == 0:
                check()
            if intent & (1 << i):
                continue
            if extent & tid == extent:
                if pivot is not None and i < pivot:
                    return None
                intent |= 1 << i
                size += 1
                if size > zmax:
                    return None
        return intent

    root = closure(all_rows, 0, None)
    if root is None:
        return
    stack = [[root, all_rows, 0, False]]
    while stack:
        check()
        intent, extent, next_item, emitted = stack[-1]
        if stats is not None:
            stats['max_dfs_depth'] = max(stats.get('max_dfs_depth',0), len(stack))
        if not emitted:
            stack[-1][3] = True
            size = intent.bit_count()
            if zmin <= size <= zmax:
                is_maximal = True
                if maximal:
                    for i,tid in enumerate(tids):
                        if not(intent & (1 << i)) and (extent & tid).bit_count() >= minimum:
                            is_maximal = False
                            break
                if is_maximal:
                    selected = []
                    bits = intent
                    while bits:
                        bit = bits & -bits
                        selected.append(item_ids[bit.bit_length()-1])
                        bits ^= bit
                    yield tuple(selected), extent.bit_count()
        if intent.bit_count() >= zmax or next_item >= count:
            stack.pop()
            continue
        stack[-1][2] += 1
        if intent & (1 << next_item):
            continue
        new_extent = extent & tids[next_item]
        if stats is not None:
            stats['extent_intersections'] = stats.get('extent_intersections',0)+1
        if new_extent.bit_count() < minimum:
            continue
        new_intent = closure(new_extent, intent, next_item)
        if new_intent is not None:
            stack.append([new_intent, new_extent, next_item+1, False])


def bounded_closed_sets(columns, transactions, minimum, zmin, zmax,
                        check=lambda: None, stats=None, row_masks=None):
    """Closed within the size-limited search space, not against larger sets.

    Below the cap, closedness is global: any equal-support extension has an
    equal-support one-item extension within the allowed space. At the cap,
    every frequent set is eligible. Both passes stream without a dedup table.
    """
    if zmax < zmin:
        return
    # The two passes have disjoint cardinalities, so duplicates cannot arise.
    if zmin < zmax:
        yield from closed_sets(columns, transactions, minimum, zmin, zmax - 1,
                               check, stats, row_masks=row_masks)
    active = [(i, extent) for i, extent in enumerate(columns)
              if extent and extent.bit_count() >= minimum]
    if transactions < minimum or len(active) < zmax:
        return

    # Iterative Eclat-style DFS: retain only projected tails along the current
    # path. There is no repository of already-emitted cap-size candidates.
    stack = [[(), active, 0]]
    while stack:
        check()
        prefix, tail, pos = stack[-1]
        need = zmax - len(prefix)
        if stats is not None:
            stats['max_boundary_dfs_depth'] = max(
                stats.get('max_boundary_dfs_depth', 0), len(stack))
        if pos > len(tail) - need:
            stack.pop()
            continue
        stack[-1][2] += 1
        item, extent = tail[pos]
        group = prefix + (item,)
        if need == 1:
            yield group, extent.bit_count()
            continue
        projected = []
        for other, other_extent in tail[pos + 1:]:
            check()
            intersection = extent & other_extent
            if stats is not None:
                stats['boundary_intersections'] = stats.get('boundary_intersections', 0) + 1
            if intersection.bit_count() >= minimum:
                projected.append((other, intersection))
        if len(projected) >= need - 1:
            stack.append([group, projected, 0])


def mine_vertical(matrix, names, min_gene_support, min_coreg_support, max_coreg,
                  max_itemsets=None, timeout_seconds=None, temp_dir=None,
                  chunk_bytes=1024*1024, _output_directory=None, spill_threshold_bytes=None):
    started = time.perf_counter()
    def check():
        if timeout_seconds is not None and (time.perf_counter() - started) >= timeout_seconds:
            raise TimeoutError('Vertical mining exceeded timeout_seconds; no partial result returned.')
    columns = bitsets_from_matrix(matrix, check)
    transactions = len(matrix)
    singleton_min = minimum_count(min_gene_support, transactions)
    coreg_min = minimum_count(min_coreg_support, transactions)
    frequent_ids = [i for i,tid in enumerate(columns) if tid and tid.bit_count() >= coreg_min]
    rows = row_bitsets(matrix, frequent_ids, check) if max_coreg > 1 and frequent_ids else None
    writer = AdaptiveCandidateWriter(names, spill_threshold_bytes, temp_dir, chunk_bytes, _output_directory)
    total = 0
    stats = {}
    def emit(indices):
        nonlocal total
        check()
        if max_itemsets is not None and total >= max_itemsets:
            raise MemoryError(f'Mining exceeded max_itemsets={max_itemsets:,}; no partial result returned.')
        writer.append(indices)
        total += 1
    try:
        for i, tid in enumerate(columns):
            if tid and tid.bit_count() >= singleton_min:
                emit((i,))
        if max_coreg > 1:
            for indices, support in bounded_closed_sets(columns, transactions, coreg_min, 2, max_coreg, check, stats, row_masks=rows):
                emit(indices)
        if not total:
            raise RuntimeError('No frequent itemsets found with the given parameters.')
        result = writer.finish()
        result.diagnostics = dict(algorithm='vertical', multi_item_target='size-bounded closed', closure_scope='max_coreg',
            storage='disk' if writer.disk is not None else 'memory', itemsets=len(result),
            candidate_file_bytes=result.buffer_bytes if writer.disk is not None else 0,
            resident_candidate_payload_bytes=result.buffer_bytes if writer.disk is None else 0,
            spill_threshold_bytes=spill_threshold_bytes,
            compact_buffer_bytes=result.buffer_bytes, chunk_bytes=chunk_bytes,
            vertical_bitsets_python_bytes=sum(sys.getsizeof(tid) for tid in columns),
            transaction_bitsets_python_bytes=sum(sys.getsizeof(row) for row in rows) if rows is not None else 0,
            seconds=time.perf_counter()-started, transactions=transactions,
            regulators=matrix.shape[1], min_coreg_occurrences=coreg_min,
            enumeration=stats, ordering='cardinality then canonical enumeration')
        return result
    except BaseException:
        writer.abort()
        raise
