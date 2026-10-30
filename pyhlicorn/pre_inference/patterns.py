"""Default streaming closed mining, optional legacy PyFIM, and monitoring."""
import math
import time
from array import array
import numpy as np
import pandas as pd
def _pyfim():
    try:
        import fim
    except ImportError as exc:
        raise ImportError("Legacy mining requires pip install 'pyhlicorn[pyfim]'. "
                          "The default vertical miner does not require PyFIM.") from exc
    return fim


class CandidateSets:
    """CSR-like integer storage; TF names are stored once, not per itemset."""
    def __init__(self, names):
        self.names = tuple(names)
        self.indices = array('I')
        self.offsets = array('Q', [0])
        self.diagnostics = {}

    def append(self, indices):
        self.indices.extend(indices)
        self.offsets.append(len(self.indices))

    def __len__(self):
        return len(self.offsets) - 1

    def iter_indices(self):
        """Read-only NumPy views; no TF-name lists during target screening."""
        buffer = np.frombuffer(self.indices, dtype=np.uint32)
        buffer.flags.writeable = False
        for i in range(len(self)):
            yield buffer[self.offsets[i]:self.offsets[i + 1]]

    def __iter__(self):
        # Preserve the low-level public iterable; inference uses iter_indices.
        for indices in self.iter_indices():
            yield indices.tolist(), [self.names[int(i)] for i in indices]

    @property
    def buffer_bytes(self):
        return len(self.indices) * self.indices.itemsize + len(self.offsets) * self.offsets.itemsize


def _mine(matrix, names, min_gene_support, min_coreg_support, max_coreg,
          algorithm='vertical', max_itemsets=None, timeout_seconds=None, temp_dir=None,
          chunk_bytes=1024*1024, _output_directory=None, spill_threshold_bytes=None):
    if algorithm == 'vertical':
        from .vertical_mining import mine_vertical
        return mine_vertical(matrix, names, min_gene_support, min_coreg_support, max_coreg,
                             max_itemsets, timeout_seconds, temp_dir, chunk_bytes, _output_directory, spill_threshold_bytes)
    fim = _pyfim()
    started = time.perf_counter()
    # Preserve all 2*n samples, including empty transactions: support denominator matters.
    transactions = [np.flatnonzero(row).tolist() for row in matrix]
    miner = getattr(fim, algorithm)
    result = CandidateSets(names)
    from .vertical_mining import bitsets_from_matrix
    columns = None
    def append_raw(raw, below_cap=False):
        nonlocal columns
        if below_cap and columns is None:
            columns = bitsets_from_matrix(matrix)
        for record in raw:
            group = tuple(sorted(record[0]))
            if below_cap:
                # Cap-size sets are enumerated exactly once by the frequent pass.
                if len(group) == max_coreg:
                    continue
                extent = (1 << len(matrix)) - 1
                for i in group:
                    extent &= columns[i]
                members = set(group)
                if any(i not in members and extent & tid == extent
                       for i, tid in enumerate(columns)):
                    continue
            if max_itemsets is not None and len(result) >= max_itemsets:
                raise MemoryError(f"Mining returned more than {max_itemsets:,} itemsets. "
                                  "No truncated network is returned. Use the RSS guard to limit native mining.")
            result.append(group)

    append_raw(miner(tracts=transactions, supp=min_gene_support * 50,
                     zmin=1, zmax=1, target='s', report=''))
    if max_coreg > 1:
        if max_coreg > 2:
            append_raw(miner(tracts=transactions, supp=min_coreg_support * 50,
                             zmin=2, zmax=max_coreg, target='c', report=''), below_cap=True)
        append_raw(miner(tracts=transactions, supp=min_coreg_support * 50,
                         zmin=max_coreg, zmax=max_coreg, target='s', report=''))
    if not result:
        raise RuntimeError('No frequent itemsets found with the given parameters.')
    result.diagnostics = dict(algorithm=algorithm, multi_item_target='size-bounded closed', closure_scope='max_coreg', report='',
                             itemsets=len(result), compact_buffer_bytes=result.buffer_bytes,
                             seconds=time.perf_counter()-started,
                             transactions=len(matrix), regulators=matrix.shape[1],
                             min_coreg_occurrences=math.ceil(min_coreg_support*len(matrix)/2))
    return result


class FrequentItemsets:
    @staticmethod
    def get_coregs(data, min_gene_support, min_coreg_support, max_coreg,
                   algorithm='vertical', memory_limit_gb=None, timeout_seconds=None,
                   max_itemsets=None, temp_dir=None, chunk_bytes=1024*1024, spill_threshold_bytes=None):
        if not all(0 <= s <= 1 for s in (min_gene_support, min_coreg_support)):
            raise ValueError('Support fractions must be between zero and one')
        if algorithm not in ('vertical', 'fpgrowth', 'apriori'):
            raise ValueError('algorithm must be vertical, fpgrowth or apriori')
        if not isinstance(max_coreg, int) or isinstance(max_coreg, bool) or max_coreg < 1:
            raise ValueError('max_coreg must be a positive integer')
        for name, value in [('memory_limit_gb', memory_limit_gb), ('timeout_seconds', timeout_seconds)]:
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')
        if max_itemsets is not None and (not isinstance(max_itemsets, int) or max_itemsets < 1):
            raise ValueError('max_itemsets must be a positive integer')
        if not isinstance(chunk_bytes, int) or chunk_bytes < 16:
            raise ValueError('chunk_bytes must be an integer >=16')
        if spill_threshold_bytes is not None and (not isinstance(spill_threshold_bytes, int) or isinstance(spill_threshold_bytes, bool) or spill_threshold_bytes < 0):
            raise ValueError('spill_threshold_bytes must be a nonnegative integer or None')
        arguments = dict(min_gene_support=min_gene_support, min_coreg_support=min_coreg_support,
                         max_coreg=max_coreg, algorithm=algorithm, max_itemsets=max_itemsets,
                         timeout_seconds=timeout_seconds, temp_dir=temp_dir, chunk_bytes=chunk_bytes,
                         spill_threshold_bytes=spill_threshold_bytes)
        matrix = data.to_numpy(dtype=bool, copy=False)
        if memory_limit_gb is not None or (timeout_seconds is not None and algorithm != 'vertical'):
            from .mining_guard import guarded_mine
            return guarded_mine(matrix, list(data.columns), arguments, memory_limit_gb, timeout_seconds)
        return _mine(matrix, list(data.columns), **arguments)

    @staticmethod
    def findMax(data, min_common_genes, min_coreg, max_coreg):
        from .vertical_mining import closed_sets
        transactions = list(data)
        names = sorted(set(name for row in transactions for name in row))
        index = {name:i for i,name in enumerate(names)}
        buffers = [bytearray((len(transactions)+7)//8) for _ in names]
        for row_number, row in enumerate(transactions):
            for name in set(row):
                buffers[index[name]][row_number//8] |= 1 << (row_number%8)
        columns = [int.from_bytes(buffer, 'little') for buffer in buffers]
        records = [([names[i] for i in indices], count/len(transactions))
                   for indices,count in closed_sets(columns, len(transactions), max(1, min_common_genes),
                                                     min_coreg, max_coreg, maximal=True)]
        return pd.DataFrame(records, columns=['itemsets','support'])[['support','itemsets']]
