"""Unit tests and regression checks for the pre-flight candidate space profiler."""

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhlicorn.pre_inference.profiler import (
    _cardinality_label,
    format_profile_table,
    profile_candidates,
)


class ProfilerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        n_genes, n_samples = 30, 20
        self.expr = pd.DataFrame(
            rng.normal(size=(n_genes, n_samples)),
            index=[f"GENE_{i}" for i in range(n_genes)],
            columns=[f"S_{j}" for j in range(n_samples)],
        )
        self.tfs = [f"GENE_{i}" for i in range(10)]

    def test_profiler_basic_output(self):
        df = profile_candidates(
            self.expr,
            self.tfs,
            sigmas=[1.0, 1.5],
            supports=[0.2, 0.3],
            max_coreg=3,
        )
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)
        expected_cols = {
            "sigma",
            "min_coreg_support",
            "eligible_tfs",
            "active_cell_pct",
            "singletons",
            "pairs",
            "triples",
            "total_candidates",
            "within_group_multitf",
            "status",
            "enumeration_seconds",
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))
        for r in df.itertuples(index=False):
            self.assertEqual(r.status, "complete")
            self.assertGreaterEqual(r.total_candidates, r.singletons)
            self.assertGreaterEqual(r.enumeration_seconds, 0.0)

    def test_profiler_supplied_discrete(self):
        disc = pd.DataFrame(
            np.where(self.expr > 0.5, 1, np.where(self.expr < -0.5, -1, 0)),
            index=self.expr.index,
            columns=self.expr.columns,
        )
        df = profile_candidates(self.expr, self.tfs, discrete=disc, supports=[0.15, 0.25])
        self.assertEqual(len(df), 2)
        self.assertTrue((df["sigma"] == "supplied").all())

    def test_regression_tf_dataframe_column_order(self):
        """Regression test: Two-column TF DataFrame must select column 0 matching inference."""
        tf_df = pd.DataFrame(
            {
                "COL_0": [f"GENE_{i}" for i in range(5)],
                "COL_1": [f"NONEXISTENT_{i}" for i in range(5)],
            }
        )
        df = profile_candidates(self.expr, tf_df, sigmas=[1.0], supports=[0.2])
        self.assertGreaterEqual(df.iloc[0]["eligible_tfs"], 2)

    def test_regression_discrete_validation_rejects_invalid_values(self):
        """Regression test: Discrete values other than -1, 0, 1 must raise ValueError."""
        disc_invalid = pd.DataFrame(
            np.zeros_like(self.expr, dtype=int),
            index=self.expr.index,
            columns=self.expr.columns,
        )
        disc_invalid.iloc[0, 0] = 2  # Invalid discrete state
        with self.assertRaises(ValueError):
            profile_candidates(self.expr, self.tfs, discrete=disc_invalid)

    def test_regression_discrete_validation_rejects_mismatched_index(self):
        """Regression test: Mismatched row/col labels must raise ValueError."""
        disc_mismatch = pd.DataFrame(
            np.zeros_like(self.expr, dtype=int),
            index=[f"DIFF_{i}" for i in range(len(self.expr))],
            columns=self.expr.columns,
        )
        with self.assertRaises(ValueError):
            profile_candidates(self.expr, self.tfs, discrete=disc_mismatch)

    def test_regression_candidate_cap_enforced_on_singletons(self):
        """Regression test: max_candidates_per_cell must cap singletons when max_coreg=1."""
        df = profile_candidates(
            self.expr,
            self.tfs,
            sigmas=[1.0],
            supports=[0.2],
            max_coreg=1,
            max_candidates_per_cell=2,
        )
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertIn("incomplete", row["status"])
        self.assertEqual(row["total_candidates"], 2)

    def test_regression_timeout_caught_cleanly(self):
        """Regression test: Timeout during bitset setup/enumeration must not escape as uncaught exception."""
        df = profile_candidates(
            self.expr,
            self.tfs,
            sigmas=[1.0],
            supports=[0.05],
            timeout_per_cell=1e-8,
        )
        self.assertEqual(len(df), 1)
        self.assertIn("incomplete (timeout", df.iloc[0]["status"])

    def test_regression_incomplete_absence_is_unknown(self):
        """Regression test: Interrupted enumeration before multi-TF sets must report within_group_multitf as None."""
        df = profile_candidates(
            self.expr,
            self.tfs,
            sigmas=[1.0],
            supports=[0.05],
            max_candidates_per_cell=1,
        )
        self.assertEqual(len(df), 1)
        self.assertIsNone(df.iloc[0]["within_group_multitf"])

    def test_regression_higher_cardinalities_reported(self):
        """Regression test: max_coreg=4 must include quadruples column and display in table."""
        df = profile_candidates(
            self.expr,
            self.tfs,
            sigmas=[1.0],
            supports=[0.1],
            max_coreg=4,
        )
        self.assertIn("quadruples", df.columns)
        table = format_profile_table(df)
        self.assertIn("Quadruples", table)

    def test_regression_parameter_validation_rejects_nan_and_invalids(self):
        """Regression test: NaN sigmas, NaN timeouts, and non-positive limits must raise ValueError."""
        with self.assertRaises(ValueError):
            profile_candidates(self.expr, self.tfs, sigmas=[float("nan")])
        with self.assertRaises(ValueError):
            profile_candidates(self.expr, self.tfs, sigmas=[-1.0])
        with self.assertRaises(ValueError):
            profile_candidates(self.expr, self.tfs, timeout_per_cell=float("nan"))
        with self.assertRaises(ValueError):
            profile_candidates(self.expr, self.tfs, timeout_per_cell=-5.0)
        with self.assertRaises(ValueError):
            profile_candidates(self.expr, self.tfs, max_candidates_per_cell=-10)
        with self.assertRaises(ValueError):
            profile_candidates(self.expr, self.tfs, max_coreg=0)

    def test_regression_zero_support_accepted(self):
        """Regression test: Support of 0.0 is accepted matching inference."""
        df = profile_candidates(self.expr, self.tfs, sigmas=[1.0], supports=[0.0])
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["min_coreg_support"], 0.0)

    def test_regression_formatting_zero_tfs_displays_exact_zero(self):
        """Regression test: Exact zero candidates displays '0', not '>=0*'."""
        # Non-matching TFs
        df = profile_candidates(self.expr, ["NONEXISTENT_TF"], sigmas=[1.0], supports=[0.2])
        self.assertEqual(df.iloc[0]["total_candidates"], 0)
        table = format_profile_table(df)
        self.assertNotIn(">=0*", table)
        self.assertIn(" 0 ", table)

    def test_profiler_formatting(self):
        df = profile_candidates(
            self.expr,
            self.tfs,
            sigmas=[1.0],
            supports=[0.2],
        )
        table_str = format_profile_table(df)
        self.assertIn("Sigma", table_str)
        self.assertIn("Support", table_str)
        self.assertIn("Singletons", table_str)
        self.assertIn("Pairs", table_str)
        self.assertIn("Triples", table_str)
        self.assertIn("Multi-TF?", table_str)
        self.assertIn("Notes:", table_str)

    def test_cli_profile_flag(self):
        root = Path(__file__).resolve().parents[1]
        cmd = [
            sys.executable,
            str(root / "run_inference.py"),
            "--expr",
            str(root / "data" / "CIT_BLCA_EXP.csv"),
            "--tf",
            str(root / "data" / "HumanTF.csv"),
            "--profile",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Pre-flight Candidate Space Profiling Mode", res.stdout)
        self.assertIn("Diagnostic profiling complete. Exiting without running inference", res.stdout)


if __name__ == "__main__":
    unittest.main()
