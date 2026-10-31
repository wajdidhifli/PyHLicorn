#!/usr/bin/env python3
"""PyHLicorn network inference runner with opt-in local memory budgeting.

Run directly from your terminal:
    python run_inference.py --demo

Or provide custom data files and options:
    python run_inference.py --expr data/CIT_BLCA_EXP.csv --tf data/HumanTF.csv --n-jobs 8
"""

import sys
from pyhlicorn.cli import main

if __name__ == "__main__":
    main()
