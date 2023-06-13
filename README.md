# MDGNN_BS
This github contains the implementation of the method proposed in MDGNN_BS paper. Part of the implementation is adopted from [TGN](https://github.com/twitter-research/tgn).



## Running the experiments

### Requirements

Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

### Dataset and Preprocessing

#### Download the public data
Download the datasets (eg. wikipedia and reddit) from
[here](http://snap.stanford.edu/jodie/) and store their csv files in a folder named
```data/```.

#### Preprocess the data
Dense `npy` format is used to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
```



### Model Training with or without PRES (more details to be added later)

Self-supervised learning using the link prediction task:
```{bash}
# TGN-attn: Supervised learning on the wikipedia dataset without PRES
python train.py --use_memory --prefix tgn-attn --n_runs 10

# TGN-attn: Supervised learning on the wikipedia dataset with PRES
python train_self_supervised.py --use_memory --prefix tgn-attn-reddit --n_runs 10 --use_pres
```
