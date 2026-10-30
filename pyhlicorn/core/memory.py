"""Opt-in memory planning; estimates are not an operating-system RSS cap."""
import math
import re
import warnings


def parse_memory_budget(value):
    """Return bytes. Integers are bytes; strings require an explicit unit."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError('memory_budget must be positive bytes or a size such as 16GB')
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        match = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)\s*', value, re.I)
        if not match:
            raise ValueError('Use a size with units, for example 16GB or 16GiB')
        unit = match[2].upper()
        powers = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3, 'TB': 4,
                  'KIB': 1, 'MIB': 2, 'GIB': 3, 'TIB': 4}
        amount = float(match[1]) * (1024 if 'I' in unit else 1000) ** powers[unit]
        if not math.isfinite(amount):
            raise ValueError('memory_budget must be finite')
        result = int(amount)
    else:
        raise ValueError('memory_budget must be integer bytes or a size with units')
    if result <= 0:
        raise ValueError('memory_budget must be positive')
    return result


def plan_memory(budget, input_bytes, candidates, samples, max_coreg, requested_jobs=1,
                candidate_bytes=0):
    """Plan local workers using worst-case screening states plus headroom.

    The estimate excludes data-dependent retained ties and final output size.
    Insufficient estimates warn and use one worker; they never discard models.
    """
    from joblib import effective_n_jobs
    budget = parse_memory_budget(budget)
    requested = effective_n_jobs(requested_jobs)
    reserved = max(256 * 1024**2, int(input_bytes) * 3 + int(candidate_bytes))
    worker = 128 * 1024**2 + int(candidates) * 2 * (5 * int(samples) + 4 * int(max_coreg) + 192)
    available = max(0, int(budget * .75) - reserved)
    jobs = max(1, min(requested, available // worker))
    fits = available >= worker
    if jobs < requested:
        warnings.warn(f'Memory budget: using {jobs} of {requested} requested local workers '
                      '(conservative screening-state estimate).', RuntimeWarning, stacklevel=2)
    if not fits:
        warnings.warn('Memory budget is below the estimated single-worker working set; '
                      'continuing with one worker and candidate spill. This is not a hard RAM cap.',
                      RuntimeWarning, stacklevel=2)
    return {'budget_bytes': budget, 'requested_workers': requested, 'effective_workers': jobs,
            'estimated_driver_reserve_bytes': reserved, 'estimated_worker_bytes': worker,
            'estimated_single_worker_fits': fits, 'output_headroom_fraction': .25,
            'hard_cap': False,
            'unbounded_components': ['retained boundary ties', 'final result frames', 'network export']}
