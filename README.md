# PyHLicorn

**PyHLicorn** is an efficient Python library for inferring cooperative Gene Regulatory Networks (GRNs) directly from transcriptomic expression data, based on the hybrid h-LICORN algorithm.

By modeling combinatorial interactions among transcription factors (TFs)—identifying co-activator and co-repressor complexes—PyHLicorn bridges discrete pattern mining and continuous expression modeling. It features vertical bitset candidate mining, compact integer representation, exact concordance pre-screening, a native C scoring kernel, and scalable multicore execution.

---

## Key Features

- **Exact Model Parity**: Mathematically reproduces the model selection of the reference R implementation (R-hLICORN / CoRegNet) while eliminating legacy bottlenecks.
- **High Performance**: Achieves up to **26.2× speedups** and over **82% peak memory reduction** on real bulk transcriptomic cohorts (e.g., TCGA breast cancer).
- **Scalable Multicore Execution**: Seamless parallelization across CPU cores via Joblib (`loky` backend) with optional distributed PySpark support.
- **Adaptive Memory Management**: Automatically manages candidate regulator pools with in-memory arrays and spill-to-disk safeguards for large combinatorial spaces.
- **Command-Line Runner**: Ready-to-use CLI script (`run_inference.py`) for automated batch processing and pipeline integration.

---

## Installation

### Prerequisites
- Python $\ge$ 3.10
- A standard C compiler (e.g. `gcc` or `clang`)
- Development headers for Python and NumPy

### Install from source
Clone the repository and install in your Python environment:

```bash
git clone https://github.com/wajdidhifli/PyHLicorn.git
cd PyHLicorn

# Compile native extensions and install in editable mode:
pip install -e .
```

### Optional Dependencies
- **Distributed Spark execution**: `pip install '.[spark]'`
- **Memory profiling & monitoring**: `pip install '.[monitor]'`

---

## Quickstart

### 1. Python API

```python
import pandas as pd
from pyhlicorn import HLicorn

# 1. Load mandatory inputs
expression = pd.read_csv('data/CIT_BLCA_EXP.csv', index_col=0)
tfs = pd.read_csv('data/HumanTF.csv', index_col=0).iloc[:, 0].tolist()

# 2. Configure HLicorn with recommended algorithmic parameters
inference = HLicorn(
    min_gene_support=0.10,     # Gene non-zero variation cutoff
    min_coreg_support=0.10,    # Minimum co-regulator support
    max_coreg=3,               # Maximum co-regulator complex size (K=3)
    search_thresh=1/3,          # Target-regulator concordance cutoff
    nGRN=100                   # Rank cutoff per target
)

# 3. Configure execution backend and scoring model
inference.setJoblib(n_jobs=4, backend='loky')  # Parallel multicore execution
inference.setLinearModel('lm')                 # Ordinary Least Squares (or 'cvlm')

# 4. Fit regulatory models (auto-discretizes if discrete matrix is omitted)
network = inference.fit(
    numerical_expression=expression,
    tf_list=tfs
)

# 5. Inspect and export results
print(network.GRN.head())
network.save_as_json('inferred_network.json')
```

### 2. Command-Line Runner (`run_inference.py`)

Run network inference directly from your terminal using the bundled reference script:

```bash
# Basic run with mandatory inputs and recommended 4 workers:
python run_inference.py \
    --expr data/CIT_BLCA_EXP.csv \
    --tf data/HumanTF.csv \
    --n-jobs 4

# Advanced run with custom outputs and fit-quality filtering:
python run_inference.py \
    --expr data/CIT_BLCA_EXP.csv \
    --tf data/HumanTF.csv \
    --max-coreg 3 \
    --min-coreg-support 0.10 \
    --n-jobs 8 \
    --min-r2 0.30 \
    --out-csv inferred_grn.csv \
    --out-json inferred_network.json.gz \
    --compact-json
```

---

## Configuration & Parameters

### 1. Mandatory Inputs

These inputs are **strictly required** for network inference:

| Parameter (Python API) | CLI Flag | Type | Description |
| :--- | :--- | :--- | :--- |
| `numerical_expression` | `--expr` | `pd.DataFrame` / Path | Continuous gene expression matrix with **genes as rows** and **samples as columns** (CSV or DataFrame). |
| `tf_list` | `--tf` | `list` / Path | Catalog of transcription factor gene symbols (list of strings, single-column text file, or CSV). |

---

### 2. Recommended Parameters

These parameters govern the combinatorial search space, regulatory biology, and parallel execution:

| Parameter (Python API) | CLI Flag | Type | Default | Recommendation & Description |
| :--- | :--- | :--- | :--- | :--- |
| `max_coreg` | `--max-coreg` | `int` | `3` | **Recommended: `2` or `3`**. Maximum size of co-regulator complexes ($K$). Values above 3 exponentially increase combinatorial search space with diminishing biological returns. |
| `min_coreg_support` | `--min-coreg-support` | `float` | `0.10` | **Recommended: `0.10`–`0.30`**. Minimum proportion of samples sharing a co-regulatory pattern. For large cohorts ($N > 200$), increasing to `0.20`–`0.30` significantly speeds up search. |
| `min_gene_support` | `--min-gene-support` | `float` | `0.10` | Minimum proportion of samples displaying non-zero discretized expression variation for a gene to be evaluated. |
| `search_thresh` | `--search-thresh` | `float` | `0.333` | Minimum concordance threshold between target gene and candidate regulator itemsets ($1/3$). |
| `nGRN` | `--n-grn` | `int` | `100` | Discrete-score rank cutoff per gene, retaining boundary ties. Retains the top candidate models before continuous OLS fitting. |
| `setJoblib(n_jobs=...)` | `--n-jobs` | `int` | `1` (API) / `8` (CLI) | **Recommended: number of CPU cores**. Number of concurrent worker processes (`-1` uses all available CPU threads). |
| `setLinearModel(...)` | `--cv` | `str` / Flag | `'lm'` | Linear model scoring mode: `'lm'` (default) fits Ordinary Least Squares; `'cvlm'` (or `--cv` in CLI) performs 10-fold cross-validated linear regression. |

---

### 3. Optional & Advanced Parameters

These options allow customizing discretization, restricting target genes, managing memory, and exporting results:

| Parameter (Python API) | CLI Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `discrete_expression` | `--disc` | `pd.DataFrame` / Path | `None` | Pre-computed ternary expression matrix $\{-1, 0, 1\}$ (under-expressed, neutral, over-expressed). If omitted, PyHLicorn auto-discretizes using `--sigma`. |
| `threshold` | `--sigma` | `float` | `1.0` | Standard deviation multiplier used for Gaussian ternary auto-discretization ($[-\sigma, +\sigma]$) when `--disc` is not provided. |
| `gene_list` | `--targets` | `list` / Path | `None` | Specific target genes for which regulatory models should be inferred. If omitted, all non-TF genes in the expression matrix are evaluated. |
| `network.filter_models(...)` | `--min-r2` | `float` | `None` | Quality filter to drop models with $R^2 < \text{threshold}$ (e.g. `0.30` or `0.40`) prior to exporting. |
| `memory_budget` | `--memory-budget` | `str / int` | `None` | Local working memory budget (e.g., `"16GB"` or integer bytes). Automatically enables candidate disk spilling and throttles worker concurrency; not a hard RSS limit. |
| `profile_candidates(...)` | `--profile` | Function / Flag | — | Diagnostic pre-flight check across multiple discretization thresholds ($\sigma$) and supports, printing a candidate count summary table and exiting without running inference. |
| `network.save_as_json(...)` | `--out-json` | Path | `inferred_network.json` | Path to save the full network model in JSON format. Automatically compresses when given a `.json.gz` filename. |
| — | `--out-csv` | Path | `inferred_grn.csv` | Path to export the inferred regulatory interactions table in CSV format. |
| `indent=None` | `--compact-json` | Flag | `False` | Omit whitespace indentation in JSON export to significantly reduce file size on disk. |

---

## Testing & Verification

The test suite validates pattern mining, mathematical equivalence, and memory policies:

```bash
pytest tests/ -v
```

---

## References & Attribution

If you use PyHLicorn in your research, please cite:

- **PyHLicorn**: Dhifli, W., Abu Awad, M., Kotob, J., Karabadji, N. E., Assi, A., & Elati, M. (2026). *PyHLicorn: efficient cooperative gene regulatory network inference in Python*. (Manuscript under review).
- **h-LICORN**: Chebil, I., Nicolle, R., Santini, G., Rouveirol, C., & Elati, M. (2014). *Hybrid method inference for the construction of cooperative regulatory network in human*. IEEE Transactions on NanoBioscience, 13(2), 97–103. DOI: 10.1109/TNB.2014.2316920.
- **LICORN**: Elati, M., Neuvial, P., Bolotin-Fukuhara, M., Barillot, E., Radvanyi, F., & Rouveirol, C. (2007). *LICORN: learning cooperative regulation networks from gene expression data*. Bioinformatics, 23(18), 2407–2414. DOI: 10.1093/bioinformatics/btm352.
- **CoRegNet**: Nicolle, R., Radvanyi, F., & Elati, M. (2015). *CoRegNet: reconstruction and integrated analysis of co-regulatory networks*. Bioinformatics, 31(18), 3066–3068. DOI: 10.1093/bioinformatics/btv305.

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
