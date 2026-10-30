import tempfile
import unittest
from pathlib import Path
import pandas as pd
from pyhlicorn.core.network import Network

class ModelFilterTests(unittest.TestCase):
    def network(self):
        return Network(pd.DataFrame([
            dict(Target='g1',coact=['A'],corep=[],R2=.9),
            dict(Target='g2',coact=['B'],corep=['C'],R2=.1),
            dict(Target='g3',coact=['D'],corep=[],R2=float('inf')),
        ]))

    def test_filter_rebuilds_counts_and_discards_derived_analyses(self):
        original=self.network();original.coregs=pd.DataFrame({'stale':[1]});original.coregsinfo={'stale':True}
        filtered=original.filter_models(.5)
        self.assertEqual(filtered.metadata['target_genes'],1)
        self.assertEqual(filtered.metadata['transcription_factors'],1)
        self.assertEqual(filtered.metadata['interactions'],1)
        self.assertEqual(set(filtered.adjlist.bygene),{'g1'})
        self.assertEqual(set(filtered.adjlist.bytf),{'A'})
        self.assertIsNone(filtered.coregs);self.assertIsNone(filtered.coregsinfo)
        self.assertEqual(len(original.GRN),3)
        self.assertNotIn('model_filter',original.metadata)
        filtered._adjlist=None
        self.assertEqual(set(filtered.adjlist.bygene),{'g1'})

    def test_empty_and_nonempty_plain_and_gzip_roundtrips(self):
        for cutoff in (.5,2):
            filtered=self.network().filter_models(cutoff)
            with tempfile.TemporaryDirectory() as directory:
                for suffix in ('.json','.json.gz'):
                    path=Path(directory)/('network'+suffix)
                    filtered.save_as_json(path,indent=None)
                    restored=Network.load_from_json(path)
                    self.assertEqual(list(restored.GRN.columns),list(filtered.GRN.columns))
                    self.assertEqual(len(restored.GRN),len(filtered.GRN))
                    self.assertEqual(restored.metadata,filtered.metadata)
                    self.assertEqual(restored.adjlist,filtered.adjlist)

    def test_invalid_threshold_or_missing_scores(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):self.network().filter_models(value)
        with self.assertRaises(ValueError):
            Network(pd.DataFrame([dict(Target='g',coact=[],corep=[])])).filter_models(.1)
        with self.assertRaises(ValueError):
            self.network().filter_models()
        for bad_k in (0, -1, 'invalid', 1.5):
            with self.assertRaises(ValueError):
                self.network().filter_models(top_k=bad_k)

    def test_best_r2_and_top_k_filtering(self):
        df = pd.DataFrame([
            dict(Target='g1', coact=['A'], corep=[], R2=0.90, RMSE=0.20),
            dict(Target='g1', coact=['B'], corep=[], R2=0.90, RMSE=0.10),  # tie-breaker best
            dict(Target='g1', coact=['C'], corep=[], R2=0.70, RMSE=0.30),
            dict(Target='g2', coact=['D'], corep=[], R2=0.60, RMSE=0.25),
            dict(Target='g2', coact=['E'], corep=[], R2=0.20, RMSE=0.50),
        ])
        net = Network(df)
        self.assertEqual(len(net.GRN), 5)

        # 1. best_r2=True retains exactly 1 model per target
        best_net = net.filter_models(best_r2=True)
        self.assertEqual(len(best_net.GRN), 2)
        self.assertEqual(set(best_net.adjlist.bygene), {'g1', 'g2'})
        self.assertEqual(best_net.metadata['model_filter']['best_r2'], True)
        self.assertEqual(best_net.metadata['model_filter']['top_k'], 1)
        # Check that g1 chose RMSE=0.10 as tie-breaker
        g1_row = best_net.GRN.loc['g1']
        self.assertEqual(g1_row['coact'], ['B'])
        self.assertEqual(g1_row['RMSE'], 0.10)

        # 2. top_k=2 retains at most 2 models per target
        top2_net = net.filter_models(top_k=2)
        self.assertEqual(len(top2_net.GRN), 4)

        # 3. Combined min_r2=0.5 and best_r2=True
        comb_net = net.filter_models(min_r2=0.5, best_r2=True)
        self.assertEqual(len(comb_net.GRN), 2)

        # 4. Combined min_r2=0.8 and best_r2=True (g2 dropped, only g1 retained)
        high_comb = net.filter_models(min_r2=0.8, best_r2=True)
        self.assertEqual(len(high_comb.GRN), 1)
        self.assertEqual(set(high_comb.adjlist.bygene), {'g1'})

    def test_rmse_filtering(self):
        df = pd.DataFrame([
            dict(Target='g1', coact=['A'], corep=[], R2=0.80, RMSE=0.15),
            dict(Target='g1', coact=['B'], corep=[], R2=0.90, RMSE=0.25),
            dict(Target='g2', coact=['C'], corep=[], R2=0.60, RMSE=0.35),
            dict(Target='g2', coact=['D'], corep=[], R2=0.50, RMSE=0.45),
        ])
        net = Network(df)

        # 1. max_rmse cutoff
        filtered_rmse = net.filter_models(max_rmse=0.25)
        self.assertEqual(len(filtered_rmse.GRN), 2)
        self.assertEqual(set(filtered_rmse.adjlist.bygene), {'g1'})
        self.assertEqual(filtered_rmse.metadata['model_filter']['max_rmse'], 0.25)

        # 2. Combined min_r2 and max_rmse
        comb = net.filter_models(min_r2=0.55, max_rmse=0.35)
        self.assertEqual(len(comb.GRN), 3)

        # 3. best_rmse=True (minimizes RMSE per target)
        best_rmse_net = net.filter_models(best_rmse=True)
        self.assertEqual(len(best_rmse_net.GRN), 2)
        # For g1, chooses RMSE=0.15 (despite lower R2 than 0.90)
        self.assertEqual(best_rmse_net.GRN.loc['g1']['coact'], ['A'])
        self.assertEqual(best_rmse_net.GRN.loc['g1']['RMSE'], 0.15)
        # For g2, chooses RMSE=0.35
        self.assertEqual(best_rmse_net.GRN.loc['g2']['coact'], ['C'])
        self.assertEqual(best_rmse_net.GRN.loc['g2']['RMSE'], 0.35)

        # 4. Error validations
        with self.assertRaises(ValueError):
            net.filter_models(max_rmse=-0.1)
        for bad_val in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                net.filter_models(max_rmse=bad_val)
        with self.assertRaises(ValueError):
            net.filter_models(best_r2=True, best_rmse=True)
        with self.assertRaises(ValueError):
            Network(pd.DataFrame([dict(Target='g', coact=[], corep=[], R2=0.5)])).filter_models(max_rmse=0.2)

    def test_infinity_scores(self):
        df = pd.DataFrame([
            dict(Target='g1', coact=['A'], corep=[], R2=0.80, RMSE=0.20),
            dict(Target='g1', coact=['B'], corep=[], R2=float('inf'), RMSE=0.10),
            dict(Target='g1', coact=['C'], corep=[], R2=-float('inf'), RMSE=0.05),
            dict(Target='g2', coact=['D'], corep=[], R2=0.70, RMSE=float('inf')),
            dict(Target='g2', coact=['E'], corep=[], R2=0.60, RMSE=0.30),
        ])
        net = Network(df)

        # min_r2 excludes +/-inf
        filtered_r2 = net.filter_models(min_r2=0.50)
        self.assertEqual(len(filtered_r2.GRN), 3)  # g1(A), g2(D), g2(E)
        self.assertNotIn('B', [m for row in filtered_r2.GRN['coact'] for m in row])

        # max_rmse excludes inf
        filtered_rmse = net.filter_models(max_rmse=0.50)
        self.assertEqual(len(filtered_rmse.GRN), 4)  # g1(A,B,C), g2(E)
        self.assertNotIn('D', [m for row in filtered_rmse.GRN['coact'] for m in row])

        # best_r2 excludes inf from primary ranking; picks g1(A) with R2=0.80
        best_r2_net = net.filter_models(best_r2=True)
        self.assertEqual(len(best_r2_net.GRN), 2)
        self.assertEqual(best_r2_net.GRN.loc['g1']['coact'], ['A'])

        # best_rmse excludes inf from primary ranking; picks g2(E) with RMSE=0.30
        best_rmse_net = net.filter_models(best_rmse=True)
        self.assertEqual(len(best_rmse_net.GRN), 2)
        self.assertEqual(best_rmse_net.GRN.loc['g2']['coact'], ['E'])

    def test_all_nan_target_groups(self):
        df = pd.DataFrame([
            dict(Target='g1', coact=['A'], corep=[], R2=0.75, RMSE=0.25),
            dict(Target='g1', coact=['B'], corep=[], R2=0.85, RMSE=0.20),
            dict(Target='g2', coact=['C'], corep=[], R2=float('nan'), RMSE=0.30),
            dict(Target='g2', coact=['D'], corep=[], R2=None, RMSE=0.40),
            dict(Target='g3', coact=['E'], corep=[], R2="not_a_number", RMSE=0.50),
        ])
        net = Network(df)

        # best_r2 excludes targets with all-NaN primary scores completely
        filtered = net.filter_models(best_r2=True)
        self.assertEqual(len(filtered.GRN), 1)
        self.assertEqual(filtered.metadata['target_genes'], 1)
        self.assertEqual(set(filtered.adjlist.bygene), {'g1'})
        self.assertEqual(filtered.GRN.loc['g1']['coact'], ['B'])

        # All-NaN network returns empty Network with 0 targets
        all_nan_df = pd.DataFrame([
            dict(Target='g1', coact=['A'], corep=[], R2=float('nan'), RMSE=0.1),
            dict(Target='g2', coact=['B'], corep=[], R2=float('nan'), RMSE=0.2),
        ])
        all_nan_net = Network(all_nan_df).filter_models(best_r2=True)
        self.assertEqual(len(all_nan_net.GRN), 0)
        self.assertEqual(all_nan_net.metadata['target_genes'], 0)
        self.assertEqual(all_nan_net.metadata['interactions'], 0)

    def test_numeric_string_scores(self):
        # Lexicographically '9.0' > '10.0', but numerically 10.0 > 9.0
        df = pd.DataFrame([
            dict(Target='g1', coact=['A'], corep=[], R2="9.0", RMSE="0.5"),
            dict(Target='g1', coact=['B'], corep=[], R2="10.0", RMSE="0.5"),
            # Lexicographically '0.1' < '0.09' is False ('0.09' < '0.1'), but numerically 0.09 < 0.1
            dict(Target='g2', coact=['C'], corep=[], R2="0.5", RMSE="0.1"),
            dict(Target='g2', coact=['D'], corep=[], R2="0.5", RMSE="0.09"),
        ])
        net = Network(df)

        best_r2_net = net.filter_models(best_r2=True)
        self.assertEqual(best_r2_net.GRN.loc['g1']['coact'], ['B'])
        self.assertAlmostEqual(float(best_r2_net.GRN.loc['g1']['R2']), 10.0)

        best_rmse_net = net.filter_models(best_rmse=True)
        self.assertEqual(best_rmse_net.GRN.loc['g2']['coact'], ['D'])
        self.assertAlmostEqual(float(best_rmse_net.GRN.loc['g2']['RMSE']), 0.09)

    def test_missing_columns_validation(self):
        # Missing R2 column
        net_no_r2 = Network(pd.DataFrame([dict(Target='g1', coact=['A'], corep=[], RMSE=0.2)]))
        with self.assertRaises(ValueError):
            net_no_r2.filter_models(min_r2=0.5)
        with self.assertRaises(ValueError):
            net_no_r2.filter_models(best_r2=True)
        with self.assertRaises(ValueError):
            net_no_r2.filter_models(top_k=2)

        # Missing RMSE column
        net_no_rmse = Network(pd.DataFrame([dict(Target='g1', coact=['A'], corep=[], R2=0.5)]))
        with self.assertRaises(ValueError):
            net_no_rmse.filter_models(max_rmse=0.2)
        with self.assertRaises(ValueError):
            net_no_rmse.filter_models(best_rmse=True)

        # Empty Network input missing columns
        empty_no_r2 = Network(pd.DataFrame(columns=['Target', 'coact', 'corep']))
        with self.assertRaises(ValueError):
            empty_no_r2.filter_models(min_r2=0.5)
        with self.assertRaises(ValueError):
            empty_no_r2.filter_models(best_r2=True)
        with self.assertRaises(ValueError):
            empty_no_r2.filter_models(best_rmse=True)

        # Empty Network with valid columns passes cleanly
        empty_with_r2 = Network(pd.DataFrame(columns=['Target', 'coact', 'corep', 'R2']))
        res = empty_with_r2.filter_models(best_r2=True)
        self.assertEqual(len(res.GRN), 0)

    def test_negative_rmse_exclusion(self):
        df = pd.DataFrame([
            # Target g1: Model A has invalid negative RMSE, Model B has valid non-negative RMSE
            dict(Target='g1', coact=['A'], corep=[], R2=0.90, RMSE=-0.50),
            dict(Target='g1', coact=['B'], corep=[], R2=0.90, RMSE=0.20),
            dict(Target='g1', coact=['C'], corep=[], R2=0.80, RMSE=0.10),
            # Target g2: only negative RMSE
            dict(Target='g2', coact=['D'], corep=[], R2=0.70, RMSE=-0.10),
        ])
        net = Network(df)

        # 1. max_rmse filtering excludes negative RMSE
        filtered_rmse = net.filter_models(max_rmse=0.30)
        self.assertEqual(len(filtered_rmse.GRN), 2)  # B and C
        self.assertNotIn('A', [m for row in filtered_rmse.GRN['coact'] for m in row])
        self.assertNotIn('D', [m for row in filtered_rmse.GRN['coact'] for m in row])

        # 2. best_rmse excludes negative RMSE from primary ranking
        best_rmse_net = net.filter_models(best_rmse=True)
        self.assertEqual(len(best_rmse_net.GRN), 1)  # only g1(C) with RMSE=0.10; g2 has no valid RMSE
        self.assertEqual(best_rmse_net.GRN.loc['g1']['coact'], ['C'])

        # 3. best_r2 with secondary RMSE excludes negative RMSE models
        best_r2_net = net.filter_models(best_r2=True)
        self.assertEqual(len(best_r2_net.GRN), 1)  # g2 dropped because its only model has negative RMSE
        self.assertEqual(best_r2_net.GRN.loc['g1']['coact'], ['B'])  # Model B chosen over A

    def test_conflicting_and_invalid_top_k(self):
        df = pd.DataFrame([dict(Target='g1', coact=['A'], corep=[], R2=0.8, RMSE=0.2)])
        net = Network(df)

        # Conflicting top_k with best flags
        with self.assertRaises(ValueError):
            net.filter_models(best_r2=True, top_k=2)
        with self.assertRaises(ValueError):
            net.filter_models(best_rmse=True, top_k=5)

        # Invalid top_k values
        for bad_k in (0, -1, -5, 1.5, "2", True, False):
            with self.assertRaises(ValueError):
                net.filter_models(top_k=bad_k)

        # Valid combinations with top_k=None or top_k=1
        res1 = net.filter_models(best_r2=True, top_k=1)
        self.assertEqual(len(res1.GRN), 1)
        res2 = net.filter_models(best_r2=True, top_k=None)
        self.assertEqual(len(res2.GRN), 1)

    def test_boolean_flags_validation(self):
        df = pd.DataFrame([dict(Target='g1', coact=['A'], corep=[], R2=0.8, RMSE=0.2)])
        net = Network(df)

        for bad_flag in (1, 0, "True", "False", [True], None):
            with self.assertRaises(ValueError):
                net.filter_models(best_r2=bad_flag)
            with self.assertRaises(ValueError):
                net.filter_models(best_rmse=bad_flag)

    def test_complete_ranking_ties_stable_sort(self):
        # Two models with completely identical R2 and RMSE
        df1 = pd.DataFrame([
            dict(Target='g1', coact=['TF1'], corep=[], R2=0.85, RMSE=0.20),
            dict(Target='g1', coact=['TF2'], corep=[], R2=0.85, RMSE=0.20),
        ])
        net1 = Network(df1)
        best1 = net1.filter_models(best_r2=True)
        self.assertEqual(best1.GRN.loc['g1']['coact'], ['TF1'])

        # Reversed order in input: TF2 is first
        df2 = pd.DataFrame([
            dict(Target='g1', coact=['TF2'], corep=[], R2=0.85, RMSE=0.20),
            dict(Target='g1', coact=['TF1'], corep=[], R2=0.85, RMSE=0.20),
        ])
        net2 = Network(df2)
        best2 = net2.filter_models(best_r2=True)
        self.assertEqual(best2.GRN.loc['g1']['coact'], ['TF2'])

        # Complete tie when secondary metric RMSE is missing (NaN) for both
        df3 = pd.DataFrame([
            dict(Target='g1', coact=['TF1'], corep=[], R2=0.85, RMSE=float('nan')),
            dict(Target='g1', coact=['TF2'], corep=[], R2=0.85, RMSE=float('nan')),
        ])
        net3 = Network(df3)
        best3 = net3.filter_models(best_r2=True)
        self.assertEqual(best3.GRN.loc['g1']['coact'], ['TF1'])

