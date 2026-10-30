#!/usr/bin/env python3
"""
PyHLicorn network inference runner with opt-in local memory budgeting

Run directly from your terminal:
    python run_inference.py

Or provide custom data files and options:
    python run_inference.py --expr path/to/expression.csv --tf path/to/tfs.csv --n-jobs 8
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
from pyhlicorn import HLicorn
from pyhlicorn.core.memory import parse_memory_budget


def load_symbol_list(path: Path) -> list:
    """Robustly load gene or TF symbols from CSV or plain text."""
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            col0_name = str(df.columns[0]).strip()
            symbols = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
            if col0_name.lower() not in {"tf", "gene", "symbol", "name", "target", "regulator", "x", "unnamed: 0"}:
                symbols = [col0_name] + symbols
            return [s for s in symbols if s]
        elif df.shape[1] > 1:
            col0_name = str(df.columns[0]).strip()
            if col0_name == "" or col0_name.lower() in {"unnamed: 0", "index"}:
                return df.iloc[:, 1].dropna().astype(str).str.strip().tolist()
            for col in df.columns:
                if col.lower() in {"tf", "gene", "symbol", "regulator", "target"}:
                    return df[col].dropna().astype(str).str.strip().tolist()
            return df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        else:
            return df.index.dropna().astype(str).str.strip().tolist()
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip().strip('"').strip("'") for line in f if line.strip()]
        if lines and lines[0].lower() in {"tf", "gene", "symbol", "name", "target", "regulator", "x"}:
            lines = lines[1:]
        return lines


def _budget_argument(value):
    try:
        return parse_memory_budget(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer Cooperative Gene Regulatory Networks using PyHLicorn (v8 size-bounded mining)"
    )
    parser.add_argument(
        "--expr",
        type=str,
        default="data/CIT_BLCA_EXP.csv",
        help="Path to numerical gene expression CSV (genes as rows, samples as columns). Default: data/CIT_BLCA_EXP.csv",
    )
    parser.add_argument(
        "--tf",
        type=str,
        default="data/HumanTF.csv",
        help="Path to transcription factor list CSV or text file. Default: data/HumanTF.csv",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Path to target genes CSV or text file. If omitted, all non-TF genes are evaluated.",
    )
    parser.add_argument(
        "--disc",
        type=str,
        default=None,
        help="Path to discrete expression CSV (-1, 0, 1). If omitted, PyHLicorn auto-discretizes.",
    )
    parser.add_argument(
        "--min-gene-support",
        type=float,
        default=0.1,
        help="Minimum proportion of samples with non-zero expression variation (default: 0.1).",
    )
    parser.add_argument(
        "--min-coreg-support",
        type=float,
        default=0.1,
        help="Minimum proportion of samples sharing co-regulatory pattern (default: 0.1).",
    )
    parser.add_argument(
        "--max-coreg",
        type=int,
        default=3,
        help="Maximum size of co-regulator complexes (default: 3).",
    )
    parser.add_argument(
        "--search-thresh",
        type=float,
        default=1 / 3,
        help="Minimum concordance between target and regulators (default: 0.333).",
    )
    parser.add_argument(
        "--n-grn",
        type=int,
        default=100,
        help="Discrete-score rank cutoff per gene, retaining boundary ties (default: 100).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8). Set 1 for sequential.",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use 10-fold cross-validated scoring ('cvlm') instead of ordinary least squares ('lm').",
    )
    parser.add_argument(
        "--max-candidate-pairs",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--memory-budget", default=None, type=_budget_argument, metavar="SIZE",
        help="Optional local working-memory budget, e.g. 16GB. Automatically enables candidate spill and limits worker concurrency; not a hard RSS cap.",
    )
    parser.add_argument(
        "--spill-mb",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Standard deviation threshold for expression discretization when --disc is omitted (default: 1.0).",
    )
    parser.add_argument(
        "--min-r2",
        type=float,
        default=None,
        help="Optional minimum R2 filter to drop weak models before export (e.g., 0.3 or 0.4).",
    )
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="Export JSON without whitespace indentation to significantly reduce file size (also supports .json.gz).",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="inferred_grn.csv",
        help="Output CSV file path for inferred regulatory interactions. Default: inferred_grn.csv",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="inferred_network.json",
        help="Output JSON file path for full network model. Default: inferred_network.json",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run fast pre-flight candidate space profiler across discretizations and supports, print diagnostic report, and exit without running inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(" PyHLicorn: Cooperative Gene Regulatory Network Inference (v8 size-bounded mining)")
    print("=" * 70)

    # 1. Load Data
    expr_path = Path(args.expr)
    tf_path = Path(args.tf)

    if not expr_path.exists():
        print(f"Error: Expression file not found: {expr_path}", file=sys.stderr)
        sys.exit(1)
    if not tf_path.exists():
        print(f"Error: TF list file not found: {tf_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading numerical expression: {expr_path}")
    num_expr = pd.read_csv(expr_path, index_col=0)
    print(f"    Dimensions: {num_expr.shape[0]:,} genes x {num_expr.shape[1]:,} samples/cells")

    print(f"[*] Loading transcription factors: {tf_path}")
    tf_list = load_symbol_list(tf_path)
    print(f"    Candidate TFs loaded: {len(tf_list):,}")

    gene_list = None
    if args.targets:
        targets_path = Path(args.targets)
        if not targets_path.exists():
            print(f"Error: Target genes file not found: {targets_path}", file=sys.stderr)
            sys.exit(1)
        gene_list = load_symbol_list(targets_path)
        print(f"[*] Explicit target genes loaded: {len(gene_list):,}")

    disc_expr = None
    if args.disc:
        disc_path = Path(args.disc)
        if not disc_path.exists():
            print(f"Error: Discrete expression file not found: {disc_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[*] Loading precomputed discrete expression: {disc_path}")
        disc_expr = pd.read_csv(disc_path, index_col=0)
    else:
        from pyhlicorn.pre_inference.preprocessing import PreProcess
        print(f"[*] Auto-discretizing numerical expression at +/- {args.sigma:.2f} sigma...")
        disc_expr = PreProcess.discretize_gene_expression(num_expr, threshold=args.sigma)
    if args.profile:
        print("\n[*] Pre-flight Candidate Space Profiling Mode (--profile)")
        from pyhlicorn import profile_candidates, format_profile_table
        
        disc_input = disc_expr if args.disc else None
        sigmas = sorted(set([args.sigma, 1.0, 1.25, 1.5])) if not args.disc else None
        supports = sorted(set([args.min_coreg_support, 0.10, 0.20, 0.30]))
        
        df_profile = profile_candidates(
            expression=num_expr,
            tfs=tf_list,
            discrete=disc_input,
            sigmas=sigmas,
            supports=supports,
            min_gene_support=args.min_gene_support,
            max_coreg=args.max_coreg,
        )
        print("\n" + format_profile_table(df_profile))
        print("\n[*] Diagnostic profiling complete. Exiting without running inference (--profile).")
        sys.exit(0)

    # 2. Configure Inference Model
    scoring_mode = "cvlm" if args.cv else "lm"
    print(f"\n[*] Configuring PyHLicorn:")
    print(f"    - min_gene_support:   {args.min_gene_support}")
    print(f"    - min_coreg_support:  {args.min_coreg_support}")
    print(f"    - max_coreg:          {args.max_coreg}")
    print(f"    - search_thresh:      {args.search_thresh:.3f}")
    print(f"    - nGRN:               {args.n_grn}")
    print(f"    - Linear model mode:  {scoring_mode}")
    max_cand_pairs = None if args.max_candidate_pairs <= 0 else args.max_candidate_pairs
    if max_cand_pairs is not None:
        print(f"    - Advanced pair guard: {max_cand_pairs:,}")
    policy = f"{args.memory_budget / 1024**3:.2f} GiB working budget" if args.memory_budget else 'RAM-first (no spill budget)'
    print(f"    - Memory policy:      {policy}")
    print(f"    - Backend:            {'Joblib (' + str(args.n_jobs) + ' cores)' if args.n_jobs != 1 else 'Sequential'}")

    inference = HLicorn(
        min_gene_support=args.min_gene_support,
        min_coreg_support=args.min_coreg_support,
        max_coreg=args.max_coreg,
        search_thresh=args.search_thresh,
        nGRN=args.n_grn,
        max_candidate_pairs=max_cand_pairs,
        verbose=True,
        memory_budget=args.memory_budget,
    )
    inference.setLinearModel(scoring_mode)
    if args.spill_mb is not None:
        inference.setMining(spill_threshold_bytes=args.spill_mb * 1024 * 1024)

    if args.n_jobs != 1:
        inference.setJoblib(n_jobs=args.n_jobs, backend="loky")
    else:
        inference.setSequential()

    # 3. Execute Inference
    print("\n[*] Starting inference...")
    t0 = time.perf_counter()
    network = inference.fit(
        numerical_expression=num_expr,
        tf_list=tf_list,
        discrete_expression=disc_expr,
        gene_list=gene_list,
        collect=True,
    )
    elapsed = time.perf_counter() - t0
    network.metadata["inference_elapsed_seconds"] = elapsed

    # 4. Report Results
    meta = network.metadata
    mining_info = meta.get("mining", {})
    storage_type = mining_info.get("storage", "memory")
    itemsets = mining_info.get("itemsets", "N/A")

    print("\n" + "=" * 70)
    print(" Inference Results (before optional model filtering)")
    print("=" * 70)
    print(f"  Wall-clock Runtime:            {elapsed:.2f} seconds")
    print(f"  Mined Co-Regulator Candidates: {itemsets:,} (Storage: {storage_type})")
    print(f"  Inferred Target Genes:         {meta.get('target_genes', 0):,}")
    print(f"  Active Transcription Factors:  {meta.get('transcription_factors', 0):,}")
    print(f"  Total Regulatory Interactions: {meta.get('interactions', 0):,}")
    print(f"  Total Retained GRN Models:     {len(network.GRN):,}")

    grn_df = network.GRN.reset_index() if "Target" not in network.GRN.columns else network.GRN.copy()

    if args.min_r2 is not None:
        network = network.filter_models(args.min_r2)
        grn_df = network.GRN.reset_index()
        print(f"\n[*] Exporting models with R2 >= {args.min_r2:.3f} (fit-quality cutoff):")
        print(f"    {len(grn_df):,} models; {network}")

    if not grn_df.empty:
        print("\n[*] Sample Top Inferred Regulatory Models:")
        cols_to_show = [c for c in ["Target", "coact", "corep", "R2", "RMSE"] if c in grn_df.columns]
        print(grn_df[cols_to_show].head(5).to_string(index=False))

    # 5. Save Outputs
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    grn_df.to_csv(out_csv, index=False)
    csv_sz_mb = out_csv.stat().st_size / (1024 * 1024)
    print(f"\n[✓] Inferred interactions exported to: {out_csv.resolve()} ({csv_sz_mb:.2f} MB)")

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.compact_json else 4
    network.save_as_json(out_json, indent=indent)
    json_sz_mb = out_json.stat().st_size / (1024 * 1024)
    print(f"[✓] Full network model exported to:    {out_json.resolve()} ({json_sz_mb:.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    main()
