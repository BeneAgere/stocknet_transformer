Applying BERTweet and transformers to StockNet

# Introduction
This repository contains the source code for running BERTweet with transformers for stock market prediction. It has two
implementions: an adaptation of the original 

# Steps to Run modified Stocknet with Transformer in Tensorflow:
1) Clone repo
2) Install conda environment from environment.yaml
3) Download embeddings from https://zenodo.org/record/3237458/files/glove.twitter.27B.50d.txt.gz?download=1 and move them under /res
4) Download dataset from https://github.com/yumoxu/stocknet-dataset and move under /data
5) Run `python Main.py`

# Steps to Run PyTorch implementation with BERTweet and transformers with cross-attention and auxiliary targets
1) Clone repo
2) Install conda environment from stocknet_torch_env.yaml
3) Download dataset from https://github.com/yumoxu/stocknet-dataset and move under /data
4) Run `python StocknetTorch.py`

Notes: 
1) Core logic for the torch implementation are in StocknetTorch.py and torch_models.py, which were written from scratch, and DataPipeTorch.py was adapted from the original stocknet.
2) Core changes to add the Transformer to Stocknet in Tensorflow can be found in Model.py

# stocknet-code

This repository is based on the original stocknet-code for stock movement prediction from tweets and historical stock 
prices releasted by: [[bib](https://aclanthology.info/papers/P18-1183/p18-1183.bib)] if you use this code,  

Yumo Xu and Shay B. Cohen. 2018. [Stock Movement Prediction from Tweets and Historical Prices](http://aclweb.org/anthology/P18-1183). In Proceedings of the 56st Annual Meeting of the Association for Computational Linguistics. Melbourne, Australia, volume 1.


## Configurations
Model configurations are listed in `config.yml` where you could set `variant_type` to *hedge, tech, fund* or *discriminative* to get four corresponding model variants, HedgeFundAnalyst, TechincalAnalyst, FundamentalAnalyst or DiscriminativeAnalyst described in the paper. 

Additionally, when you set `variant_type=hedge, alpha=0`, you would acquire IndependentAnalyst without any auxiliary effects. 

## Running

After configuration, use `s` in your terminal to start model learning and test the model after the training is completed. If you would like to do them separately, simply comment out `exe.train_and_dev()` or `exe.restore_and_test()` in `Main.py`.

