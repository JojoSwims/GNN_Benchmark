"""
Model submission contract for GNN Benchmark.

To submit a model for benchmarking, subclass ``BenchmarkModel`` and implement
``fit`` and ``predict``.  Then pass your model instance to
``BenchmarkRunner.run()``.

Example::

    from gnn_benchmark.models import BenchmarkModel, TrainingHistory

    class MyGNN(BenchmarkModel):
        name = "MyGNN"

        def fit(self, train_loader, val_loader, adj, config):
            # Train your model here
            ...
            return TrainingHistory(train_loss=[...], val_loss=[...])

        def predict(self, test_loader, adj, config):
            # Return np.ndarray of shape [num_test_samples, seq_out_len, N, D_out]
            ...
"""

from gnn_benchmark.models.base import BenchmarkModel, TrainingHistory
from gnn_benchmark.models.last_value import LastValueModel

__all__ = [
    "BenchmarkModel",
    "TrainingHistory",
    "LastValueModel",
]
