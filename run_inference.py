#!/usr/bin/env python3
"""PyHLicorn network inference runner with opt-in local memory budgeting.

Run directly from your terminal:
    python run_inference.py --demo

Or provide custom data files and options:
    python run_inference.py --expr data/CIT_BLCA_EXP.csv --tf data/HumanTF.csv --n-jobs 8
"""

import sys
from pathlib import Path

# When running from a checkout where C extensions were not compiled in-place,
# prefer the installed package in site-packages over the uncompiled local directory.
root_dir = str(Path(__file__).resolve().parent)
if root_dir in sys.path:
    c_exts = (
        list(Path(root_dir, "pyhlicorn", "cpylibs").glob("comblicorn*.so"))
        + list(Path(root_dir, "pyhlicorn", "cpylibs").glob("comblicorn*.pyd"))
    )
    if not c_exts:
        sys.path.remove(root_dir)
        sys.path.append(root_dir)

from pyhlicorn.cli import main

if __name__ == "__main__":
    main()
