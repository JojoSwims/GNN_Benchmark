We're building a pipeline to easily input datasets and feed them into various GNN for forecasting models.
To do this we build an intermediate representation layer for datasets. 
This allows us to only need one function per model/dataset (intermediate=>Model) and (dataset=>Intermediate)
The intermediate layer is composed of the relevant csv files and a JSON file describing them.



For inspiration here is a pipeline for temporal graph benchmark: https://docs.tgb.complexdatalab.com/api/tgb.linkproppred/
