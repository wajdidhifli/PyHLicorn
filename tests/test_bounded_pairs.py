"""Selection parity at block boundaries, exact ties and allocation bounds."""
import importlib
import unittest
from unittest.mock import patch
import numpy as np
from scipy.stats import rankdata

kernel = importlib.import_module('pyhlicorn.during_inference.__comblicorn')


class BoundedPairsTests(unittest.TestCase):
    def test_matches_full_kernel_for_blocks_and_rank_cutoffs(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            acts = [[0], [1, 2], [3], [2, 4], []]
            reps = [[1], [2], [3, 4], [5], [0, 4], []]
            a = rng.integers(-1, 2, (len(acts), 13), dtype=np.int32)
            r = rng.choice(np.array([-1, 0, 2], np.int32), (len(reps), 13))
            a[-1] = 0; r[-1] = 0
            g = rng.integers(-1, 2, 13, dtype=np.int32)
            scores, ai, ri = kernel.score_disjoint(a.ravel(), acts, r.ravel(), reps, g)
            ranks = rankdata(scores, method='min')
            for n in (None, 0, 1, 2, 3, 10, 100):
                keep = ranks <= (1 if n is None else n)
                for block in (1, 4, 6, 13, 100):
                    aa, rr = kernel.select_disjoint(a.ravel(), acts, r.ravel(), reps, g, n, block)
                    np.testing.assert_array_equal(aa, ai[keep])
                    np.testing.assert_array_equal(rr, ri[keep])

    def test_all_ties_retained_and_native_buffers_bounded(self):
        acts = [[i] for i in range(21)] + [[]]
        reps = [[i+30] for i in range(17)] + [[]]
        a = np.zeros((len(acts), 8), np.int32)
        r = np.zeros((len(reps), 8), np.int32)
        seen = []
        original = kernel.score_disjoint
        def checked(av, ag, rv, rg, gene):
            seen.append(len(ag)*len(rg))
            return original(av, ag, rv, rg, gene)
        diagnostics = {}
        with patch.object(kernel, 'score_disjoint', side_effect=checked):
            ai, ri = kernel.select_disjoint(a.ravel(), acts, r.ravel(), reps,
                np.zeros(8, np.int32), 1, block_pairs=19, diagnostics=diagnostics)
        self.assertLessEqual(max(seen), 19)
        self.assertEqual(len(ai), len(acts)*len(reps)-1)
        self.assertEqual(diagnostics['selected_pairs'], len(ai))
        self.assertEqual(list(zip(ai,ri)), [(i,j) for i in range(len(acts))
            for j in range(len(reps)) if i != len(acts)-1 or j != len(reps)-1])

    def test_no_valid_pairs(self):
        ai, ri = kernel.select_disjoint(np.zeros(5,np.int32), [[0]],
            np.zeros(5,np.int32), [[0]], np.zeros(5,np.int32), 100, block_pairs=1)
        self.assertEqual(len(ai), 0)
        self.assertEqual(len(ri), 0)
