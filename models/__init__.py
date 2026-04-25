"""
Model submission contract for GNN Benchmark.

To submit a model for benchmarking, subclass ``BenchmarkModel`` and implement
``fit`` and ``predict``.  Then pass your model instance to
``BenchmarkRunner.run()``.

Example::

    from gnn_benchmark.models import BenchmarkModel, TrainingHistory

    class MyGNN(BenchmarkModel):
        name = "MyGNN"

        def fit(self, x_train, y_train, x_val, y_val, adj, config):
            # x_train: (S, L, N, D_in) torch.Tensor -- NaN = missing
            ...
            return TrainingHistory(train_loss=[...], val_loss=[...])

        def predict(self, x_test, adj, config):
            # Return np.ndarray of shape [S_test, seq_out_len, N, D_out]
            ...
"""

from gnn_benchmark.models.base import BenchmarkModel, TrainingHistory
from gnn_benchmark.models.gwn import GWNConfig, GWNModel
from gnn_benchmark.models.last_value import LastValueModel
from gnn_benchmark.models.mtgnn import MTGNNConfig, MTGNNModel
from gnn_benchmark.models.mtgode import MTGODEConfig, MTGODEModel
from gnn_benchmark.models.staeformer import STAEFormerConfig, STAEFormerModel
from gnn_benchmark.models.d2stgnn import D2STGNNConfig, D2STGNNModel
from gnn_benchmark.models.gts import GTSConfig, GTSModel

__all__ = [
    "BenchmarkModel",
    "TrainingHistory",
    "LastValueModel",
    "STAEFormerModel",
    "STAEFormerConfig",
    "GWNModel",
    "GWNConfig",
    "MTGNNModel",
    "MTGNNConfig",
    "MTGODEModel",
    "MTGODEConfig",
    "D2STGNNModel",
    "D2STGNNConfig",
    "GTSModel",
    "GTSConfig",
]
