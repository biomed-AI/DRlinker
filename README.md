#DRlinker

This is the code for the "DRlinker: Deep Reinforcement Learning for optimization in fragment linking Design".

To implement our models we were based on [OpenNMT-py (v0.4.1)](http://opennmt.net/OpenNMT-py/).

## Install requirements

Create a new conda environment:

```bash
conda env create -f environment.yml
conda activate DRlinker
```

## Pre-processing 

The tokenized datasets can be found on the `data/` folder. 

We use the same datasets as [SyntaLinker](https://github.com/YuYaoYang2333/SyntaLinker), the data was originated from the ChEMBL database.

We use a shared vocabulary. The `vocab_size` and `seq_length` are chosen to include the whole datasets.


## Pre-training

Pre-training can be started by running the `training.sh` script using ChEMBL dataset

The pre-trained models can be found on the `checkpoints/` folder.

## Notice

Because the file is too large to upload, we will publish all the data and pre-trained models on MultCloud. 

Please click the [link](https://share.multcloud.link/share/b4c9bb72-a012-43ad-baec-4070f61a013b) to download and replace the original `data/` and `checkpoints/` folders.


## Fine-tuning and Testing

fine-tuning script `fine-tuning-test.sh` can be run after pre-training using ChEMBL testing dataset or real drug case.

The final predictions can be found on the `case/` folder.


## RL (beam search & Multinomial sampling)

There are two strategies to sample for the fine-tuning and testing stages, training in beam search can use train_type as B and in multinomial sampling can use train_type as M.

In most of cases, multinomial sampling performs better because of its ability to explore larger chemical space.

The input of cases can refer to case/

To train the RL model use the `train_agent_ms.py` script:

Model training of beam search can be started by running the `train_agent_ms.sh` script.


## Example

`bash fine-tuning-test.sh`




