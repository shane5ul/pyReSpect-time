"""Regenerate the regression baselines in tests/baseline/.

Run from the repository root after an intentional change to the numbers:

    python tests/make_baseline.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))          # test_regression
sys.path.insert(0, str(Path(__file__).parent.parent))   # the package itself

from test_regression import BASELINE, CASES, run_case, tag  # noqa: E402

if __name__ == "__main__":
    warnings.simplefilter("ignore")
    BASELINE.mkdir(exist_ok=True)
    for name, kwargs in CASES:
        out = BASELINE / f"{tag(name, kwargs)}.npz"
        np.savez(out, **run_case(name, kwargs))
        print("wrote", out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out)
