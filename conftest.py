"""
Pytest configuration for the GNN Benchmark test suite.

The repo directory is named ``GNN_Benchmark`` but all source imports use the
package name ``gnn_benchmark``.  The package is made importable via
``pyproject.toml`` which maps ``gnn_benchmark`` to the repo root.

One-time setup (install in editable mode)::

    pip install -e /path/to/GNN_Benchmark

The repo root is also added to ``sys.path`` here so that ``benchmark.py``
at the repo root is directly importable during tests.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).parent.resolve()   # /home/user/GNN_Benchmark
_parent = _repo_root.parent                    # /home/user/

# benchmark.py lives at the repo root
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# gnn_benchmark symlink lives one level up
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
