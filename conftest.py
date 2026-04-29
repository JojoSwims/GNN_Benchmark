"""
Pytest configuration for the GNN Benchmark test suite.

The package source lives under ``gnn_benchmark/`` at the repo root and can be
imported directly once the repo root is on ``sys.path``.  For a fully isolated
install you can also run::

    pip install -e /path/to/GNN_Benchmark
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).parent.resolve()   # /home/user/GNN_Benchmark

if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
