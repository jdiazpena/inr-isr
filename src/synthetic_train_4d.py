# -*- coding: utf-8 -*-
"""
src/synthetic_train_4d.py

Flat wrapper script for 4D synthetic SIREN model training benchmark.
Invokes benchmarks/train_synthetic_4d.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add benchmarks and src to sys.path
benchmarks_dir = Path(__file__).resolve().parent.parent / "benchmarks"
src_dir = Path(__file__).resolve().parent

if str(benchmarks_dir) not in sys.path:
    sys.path.insert(0, str(benchmarks_dir))
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from train_synthetic_4d import main, run_synthetic_4d_training

if __name__ == "__main__":
    main()
