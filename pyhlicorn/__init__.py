"""
pyhlicorn - Python's Hybrid Learning inference for Cooperative Regulatory network.

This package aims at inferring a large scale transcription co-regulatory network
from transcriptomic data and at integrating external data on gene regulation to infer and 
analyze transcriptional programs.
"""

from .core import HLicorn, Network
from .core.hlicorn import hlicorn
from .pre_inference import PreProcess, profile_candidates, format_profile_table

__all__ = ['HLicorn', 'PreProcess', 'Network', 'hlicorn', 'profile_candidates', 'format_profile_table']

