"""Dyn-GWN backbone (Ibrahim et al., ICAIF 2023).

Only the model module is used by the benchmark; the upstream trainer,
data loaders, and METR-LA / stock-volatility scripts have been removed.
The benchmark wrapper lives at :mod:`gnn_benchmark.models.dyngwn`.
"""

from gnn_benchmark.models.DynGWN.model import dyngwn

__all__ = ["dyngwn"]
