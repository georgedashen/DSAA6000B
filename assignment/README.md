# DSAA6000B assignment
Author: Zhuoyang CHEN

Most part of codes are borrowed from [Tang, Jianheng, et al. "Rethinking graph neural networks for anomaly detection." International Conference on Machine Learning. PMLR, 2022.](https://arxiv.org/abs/2205.15508)

The GCN models are directly constructed using `GCNConv` Module of pytorch version. Detail implementation can be checked at Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)

GCN model definitions and some useful functions for processing data is included in `utils.py` and `dataset.py`, respectively.

## Requirements
* pytorch
* networkx
* CUDA environment: A GPU with 24G memory

## Data

T-finance downloaded from [google drive](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing).

## Exploration 1
`python main.py --hid_dim 16 --n_layer 5 --epoch 750`

`--hid_dim` is the hidden_dimension in convolutional layers when layer>1, and `--n_layer` is the number of convolutional layers.

For investigating different combinations of hidden dimension and convolutional layers, run `sh experiment_set.sh` to reproduce the results.

## Exploration 2
`python evaluate_knn.py --k 3 --t 0.1 --epoch 750`

`--k` is the number of nearest neighbors for each node, and `--t` is the minimum threshold for cosine similarity between nodes to consider connected.

For investigating different k and t, run `sh experiment_set2.sh` to reproduce the results.

## Results
`AUC` and `average_training_time_per_epoch` are recorded for each section and results can be seen in metrics.png and time.png, respectively.

One may want to reproduce the figures by using the R script `plot.R`.
