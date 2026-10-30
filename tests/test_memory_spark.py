"""Local tests for mining equivalence and Spark worker logic (no JVM needed)."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import fim
except ImportError:
    fim = None
from pyhlicorn import Network
from pyhlicorn._serialization import native, normalize_frame
from pyhlicorn.pre_inference.patterns import FrequentItemsets
from pyhlicorn.core.oneGeneHlicorn import oneGeneHlicorn
from pyhlicorn.core.spark_backend import infer_partition, result_schema


class MemorySparkTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(7)
        reg = rng.choice([-1, 0, 1], size=(6, 20))
        bit = np.concatenate([reg == 1, reg == -1], axis=1).T
        frame = pd.DataFrame(bit, columns=list('abcdef'))
        return reg, bit, frame

    @unittest.skipIf(fim is None, 'optional reference PyFIM not installed')
    def test_compact_mining_preserves_bounded_candidates_and_names(self):
        reg, bit, frame = self.fixture()
        columns = frame.columns.copy()
        for threshold in (.1, .3):
            for algorithm in ('fpgrowth', 'apriori'):
                actual = FrequentItemsets.get_coregs(frame, .1, threshold, 3, algorithm=algorithm)
                transactions = [np.flatnonzero(row).tolist() for row in bit]
                miner = getattr(fim, algorithm)
                reference = miner(transactions, supp=5, zmax=1)
                frequent = miner(transactions, supp=threshold*50, zmin=2, zmax=3, target='s')
                for group, support in frequent:
                    extent=np.all(bit[:,list(group)],axis=1)
                    if len(group)==3 or set(np.flatnonzero(np.all(bit[extent],axis=0)))==set(group):
                        reference.append((group,support))
                self.assertEqual({frozenset(a) for a, _ in actual}, {frozenset(a) for a, _ in reference})
                self.assertEqual([names for a, names in actual], [[frame.columns[i] for i in a] for a, _ in actual])
        self.assertTrue(frame.columns.equals(columns))

    def test_partition_matches_local_and_preserves_shared_arrays(self):
        reg, bit, frame = self.fixture()
        items = FrequentItemsets.get_coregs(frame, .1, .1, 2)
        numerical = reg.T.astype(float) + np.random.default_rng(1).normal(size=reg.T.shape)*.1
        for arr in (reg, bit, numerical):
            arr.flags.writeable = False
        for model in (None, 'lm', 'cvlm'):
            folds = np.arange(20) % 5 if model == 'cvlm' else None
            state = (reg, items, bit, numerical, .3, 2, model, folds, 100000)
            rows = [('target', reg[0], numerical[:, 0])]
            expected = oneGeneHlicorn('target', reg[0], reg, items, bit, .3,
                                      numerical, numerical[:, 0], 2, model, False, cv_folds=folds)
            expected = [tuple(native(value) for value in row) for row in
                        normalize_frame(expected).itertuples(index=False, name=None)]
            self.assertEqual(list(infer_partition(iter(rows), state)), expected)

    def test_pair_guard_fails_before_cartesian_allocation(self):
        reg, bit, frame = self.fixture()
        items = FrequentItemsets.get_coregs(frame, .1, .1, 2)
        with self.assertRaisesRegex(MemoryError, 'candidate pairs'):
            oneGeneHlicorn('target', reg[0], reg, items, bit, 0, reg.T,
                          reg[0], 2, None, False, max_candidate_pairs=1)

    def test_json_roundtrip_arrays_and_missing(self):
        frame = pd.DataFrame(dict(Target=['g'], coact=[np.array(['a','b'])],
                                  corep=[pd.NA], Coef_Acts=[np.array([.1,.2])],
                                  Coef_Reps=[None], R2=[np.nan]))
        net = Network(frame)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'network.json'
            net.save_as_json(path)
            raw = json.loads(path.read_text())
            self.assertIsNone(raw['GRN'][0]['R2'])
            restored = Network.load_from_json(path)
            self.assertEqual(restored.GRN.iloc[0].coact, ['a','b'])
            self.assertEqual(restored.GRN.iloc[0].corep, [])
            self.assertEqual(restored.get_regulators('g')['act'], net.get_regulators('g')['act'])

    def test_guarded_mining_matches_direct_without_rss_monitor(self):
        _, _, frame = self.fixture()
        direct = FrequentItemsets.get_coregs(frame, .1, .1, 2)
        guarded = FrequentItemsets.get_coregs(frame, .1, .1, 2, timeout_seconds=30)
        self.assertEqual(list(direct), list(guarded))

    def test_timeout_stops_worker(self):
        _, _, frame = self.fixture()
        with self.assertRaises(TimeoutError):
            FrequentItemsets.get_coregs(frame, .1, .1, 2, timeout_seconds=.000001)

    def test_itemset_limit_returns_no_partial_result(self):
        _, _, frame = self.fixture()
        with self.assertRaises(MemoryError):
            FrequentItemsets.get_coregs(frame, .1, .1, 2, max_itemsets=1)

    def test_target_without_models_keeps_columns(self):
        reg = np.ones((5, 20), dtype=int)
        bit = np.concatenate([reg == 1, reg == -1], axis=1).T
        gene = np.tile([-1, 1], 10)
        items = (([0], ['a']),)
        for model in (None, 'lm', 'cvlm'):
            out = oneGeneHlicorn('empty', gene, reg, items, bit, 1,
                                 reg.T, gene, 2, model, False)
            self.assertTrue(out.empty)
            self.assertTrue({'Target', 'coact', 'corep'} <= set(out.columns))
            state = (reg, items, bit, reg.T, 1, 2, model, None, 100)
            self.assertEqual(list(infer_partition(iter([('empty', gene, gene)]), state)), [])

    def test_empty_partition_and_schema(self):
        try:
            schema = result_schema('lm')
        except ImportError:
            self.skipTest('optional pyspark not installed')
        self.assertEqual(schema['Coef_coActs'].dataType.simpleString(), 'double')
        self.assertEqual(schema['Coef_Acts'].dataType.simpleString(), 'array<double>')
        self.assertEqual(list(infer_partition(iter([]), (None,)*9)), [])

if __name__ == '__main__':
    unittest.main()
