# PyHLicorn

[![PyPI version](https://img.shields.io/pypi/v/pyhlicorn.svg)](https://pypi.org/project/pyhlicorn/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyhlicorn.svg)](https://pypi.org/project/pyhlicorn/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**PyHLicorn** is an efficient Python library for inferring cooperative Gene Regulatory Networks (GRNs) directly from transcriptomic expression data, based on the hybrid h-LICORN algorithm.

By modeling combinatorial interactions among transcription factors (TFs)—identifying co-activator and co-repressor complexes—PyHLicorn bridges discrete pattern mining and continuous expression modeling. It features vertical bitset candidate mining, compact integer representation, exact concordance pre-screening, a native C scoring kernel, and scalable multicore execution.

---

## Key Features

- **Exact Model Parity**: Mathematically reproduces the model selection of the reference R implementation (R-hLICORN / CoRegNet) while eliminating legacy bottlenecks.
- **High Performance**: Achieves up to **26.2× speedups** and over **82% peak memory reduction** on real bulk transcriptomic cohorts (e.g., TCGA breast cancer).
- **Scalable Multicore Execution**: Seamless parallelization across CPU cores via Joblib (`loky` backend) with optional distributed PySpark support.
- **Adaptive Memory Management**: Automatically manages candidate regulator pools with in-memory arrays and spill-to-disk safeguards for large combinatorial spaces.
- **Command-Line Interface**: Installed `pyhlicorn` command and bundled `run_inference.py` script for automated batch processing and pipeline execution.
- **Self-Contained Demo**: Packaged example dataset accessible via Python (`pyhlicorn.load_example_data()`) or CLI (`pyhlicorn --demo`).

---

## Installation

### From PyPI (Recommended)

Install the released package directly via `pip`:

```bash
pip install pyhlicorn
```

Pre-compiled binary wheels are distributed for common architectures (Linux x86_64, macOS Apple Silicon, macOS Intel, and Windows).

#### Compiler Requirements for Source Distributions
When installing from source (`.tar.gz`) or if a pre-compiled wheel is unavailable for your specific platform/Python version, `pip` compiles the high-performance C scoring kernel during installation. This requires a standard C compiler and Python development headers:
- **Linux (Ubuntu/Debian)**: `sudo apt-get install build-essential python3-dev`
- **Linux (RHEL/CentOS/Fedora)**: `sudo dnf install gcc python3-devel`
- **macOS**: `xcode-select --install`
- **Windows**: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (MSVC 14.0 or newer)

### From Source (Development)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/wajdidhifli/PyHLicorn.git
cd PyHLicorn
pip install -e .
```

### Optional Dependencies
- **Distributed PySpark execution**: `pip install 'pyhlicorn[spark]'`
- **Memory profiling & monitoring**: `pip install 'pyhlicorn[monitor]'`

---

## Quickstart

### 1. Python API

```python
import pyhlicorn

# 1. Load data (use bundled example dataset, or pass custom pandas DataFrames/CSVs)
expression, tfs = pyhlicorn.load_example_data()

# 2. Configure HLicorn with recommended algorithmic parameters
inference = pyhlicorn.HLicorn(
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

### 2. Command-Line Interface (`pyhlicorn`)

When installed via `pip`, the `pyhlicorn` command is added directly to your environment:

```bash
# Instant verification on bundled demonstration data:
pyhlicorn --demo

# Basic run with your own expression and TF files:
pyhlicorn \
    --expr path/to/expression.csv \
    --tf path/to/tfs.txt \
    --n-jobs 4

# Advanced run with custom outputs and fit-quality filtering:
pyhlicorn \
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

*(Note: In a cloned repository checkout, `python run_inference.py` provides the identical CLI functionality.)*

---

## Configuration & Parameters

### 1. Mandatory Inputs

These inputs are **strictly required** for network inference (unless using `--demo`):

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
| — | `--demo` | Flag | `False` | Run network inference on the bundled example dataset (100 genes, 20 TFs). |
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

- **PyHLicorn**: Dhifli, W., AbuAwad, M., Kotob, J., Karabadji, N. E., Assi, A., & Elati, M. (2026). *PyHLicorn: efficient cooperative gene regulatory network inference in Python*. (Manuscript under review).
- **h-LICORN**: Chebil, I., Nicolle, R., Santini, G., Rouveirol, C., & Elati, M. (2014). *Hybrid method inference for the construction of cooperative regulatory network in human*. IEEE Transactions on NanoBioscience, 13(2), 97–103. DOI: 10.1109/TNB.2014.2316920.
- **LICORN**: Elati, M., Neuvial, P., Bolotin-Fukuhara, M., Barillot, E., Radvanyi, F., & Rouveirol, C. (2007). *LICORN: learning cooperative regulation networks from gene expression data*. Bioinformatics, 23(18), 2407–2414. DOI: 10.1093/bioinformatics/btm352.
- **CoRegNet**: Nicolle, R., Radvanyi, F., & Elati, M. (2015). *CoRegNet: reconstruction and integrated analysis of co-regulatory networks*. Bioinformatics, 31(18), 3066–3068. DOI: 10.1093/bioinformatics/btv305.

---

## Authors & Affiliations

- **Wajdi Dhifli** (`wajdi.dhifli@univ-lille.fr`) — *Maintainer*  
  Univ. Lille, Inserm, CHU Lille, CNRS, U1366-UMR9020 – CRCLille – Cancer Research Center of Lille, F-59000 Lille, France

- **Mohammad AbuAwad** (`202410475@uopstd.edu.jo`)  
  University of Petra, Faculty of Information Technology, 961343 Amman, Jordan

- **Jeeda Kotob** (`jk8835@rit.edu`)  
  Rochester Institute of Technology, Dubai, United Arab Emirates

- **Nour Elislem Karabadji** (`n.karabadji@ensti-annaba.dz`)  
  National Higher School of Technology and Engineering, Laboratoire De Technologies Des Systemes Energetiques (LTSE), 23005 Annaba, Algeria  
  Electronic Document Management Laboratory (LabGED), Badji Mokhtar-Annaba University, 23000 Annaba, Algeria

- **Ali Assi** (`Ali.Assi@rit.edu`)  
  Rochester Institute of Technology, Dubai, United Arab Emirates

- **Mohamed Elati** (`mohamed.elati@univ-lille.fr`)  
  Univ. Lille, Inserm, CHU Lille, CNRS, U1366-UMR9020 – CRCLille – Cancer Research Center of Lille, F-59000 Lille, France

---

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
