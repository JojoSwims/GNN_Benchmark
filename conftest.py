"""
Pytest configuration for the GNN Benchmark test suite.

The repo directory is named ``GNN_Benchmark`` but all source imports use the
package name ``gnn_benchmark``.  A symlink at ``/home/user/gnn_benchmark``
pointing to this directory makes the package importable; the parent directory
is added to ``sys.path`` here so both ``benchmark.py`` and ``gnn_benchmark``
are discoverable.

To set up the symlink (one-time, already in place in the dev environment)::

    ln -sf /path/to/GNN_Benchmark /path/to/gnn_benchmark
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
