"""Tests for CLI entry point, example data loading, and default max_coreg."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

import pyhlicorn
from pyhlicorn import HLicorn, hlicorn, load_example_data
from pyhlicorn.cli import main, parse_args


def test_load_example_data():
    expr, tfs = load_example_data()
    assert isinstance(expr, pd.DataFrame)
    assert expr.shape == (100, 183)
    assert isinstance(tfs, list)
    assert len(tfs) == 20
    assert all(isinstance(tf, str) for tf in tfs)
    # Check that TFs exist in expression matrix
    for tf in tfs:
        assert tf in expr.index


def test_default_max_coreg_is_three():
    model = HLicorn()
    assert model._thresholds["max_coreg"] == 3

    args = parse_args([])
    assert args.max_coreg == 3


def test_cli_demo_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = Path(tmpdir) / "test_grn.csv"
        out_json = Path(tmpdir) / "test_net.json"

        # Run demo through CLI main with custom output paths and sequential execution for speed
        main([
            "--demo",
            "--n-jobs", "1",
            "--n-grn", "10",
            "--out-csv", str(out_csv),
            "--out-json", str(out_json),
        ])

        assert out_csv.exists()
        assert out_json.exists()
        assert out_csv.stat().st_size > 0
        assert out_json.stat().st_size > 0

        df = pd.read_csv(out_csv)
        assert "Target" in df.columns or "target" in df.columns or len(df) > 0
