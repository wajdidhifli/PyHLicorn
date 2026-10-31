"""
pyhlicorn - Python's Hybrid Learning inference for Cooperative Regulatory network.

This package aims at inferring a large scale transcription co-regulatory network
from transcriptomic data and at integrating external data on gene regulation to infer and 
analyze transcriptional programs.
"""

from .core import HLicorn, Network
from .core.hlicorn import hlicorn
from .pre_inference import PreProcess, profile_candidates, format_profile_table


def load_example_data():
    """Load bundled demonstration gene expression matrix and transcription factor list.

    Returns
    -------
    expression : pd.DataFrame
        Gene expression matrix (100 genes x 183 samples).
    tfs : list of str
        List of 20 candidate transcription factor symbols.

    Examples
    --------
    >>> import pyhlicorn
    >>> expr, tfs = pyhlicorn.load_example_data()
    >>> model = pyhlicorn.HLicorn(max_coreg=3)
    >>> net = model.fit(expr, tfs)
    >>> print(net.GRN.head())
    """
    import importlib.resources as pkg_resources
    import pandas as pd

    data_dir = pkg_resources.files("pyhlicorn").joinpath("data")
    expr_path = data_dir.joinpath("example_expression.csv")
    tf_path = data_dir.joinpath("example_tfs.txt")

    with expr_path.open("r", encoding="utf-8") as f:
        expr = pd.read_csv(f, index_col=0)
    with tf_path.open("r", encoding="utf-8") as f:
        tfs = [line.strip() for line in f if line.strip()]

    return expr, tfs


__all__ = [
    'HLicorn',
    'PreProcess',
    'Network',
    'hlicorn',
    'profile_candidates',
    'format_profile_table',
    'load_example_data',
]

