# Dyn-GWN: Time-Series Forecasting using Time-varying Graphs

Reference: Ibrahim et al., *Dyn-GWN: Time-Series Forecasting using
Time-varying Graphs with Applications to Finance and Traffic
Prediction*, ICAIF 2023. Dyn-GWN extends [Graph
WaveNet](https://arxiv.org/abs/1906.00121) with a time-varying graph
branch alongside the static-graph branch.

## What lives here

- `model.py` — the cleaned-up `dyngwn` `nn.Module`. Used by the
  benchmark wrapper at `gnn_benchmark.models.dyngwn:DynGWNModel`.
- `LICENSE` — upstream license, retained.

The upstream training script, METR-LA / stock-volatility data loaders,
and the in-loop partial-correlation graph builder are not part of this
benchmark and have been removed. The dynamic graphs are supplied by the
dataset loaders (`divvy-bikeshare-dynamic`, `eu-load-dynamic`,
`eu-load-both`, `divvy-bikeshare-both`) and pre-processed by the
wrapper before reaching the model.

## Citation

```bibtex
@inproceedings{Ibrahim2023,
    author = {Ibrahim, Shibal and Tell, Max and Mazumder, Rahul},
    title = {Dyn-GWN: Time-Series Forecasting using Time-varying Graphs with Applications to Finance and Traffic Prediction},
    year = {2023},
    publisher = {Association for Computing Machinery},
    booktitle = {4th ACM International Conference on AI in Finance},
    series = {ICAIF '23}
}
```
