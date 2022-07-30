#!/usr/bin/env bash


dataset_name=ChEMBL
python preprocess.py -train_src data/${dataset_name}/N/src-train \
                     -train_tgt data/${dataset_name}/N/tgt-train \
                     -valid_src data/${dataset_name}/N/src-val \
                     -valid_tgt data/${dataset_name}/N/tgt-val \
                     -save_data data/${dataset_name}/N/${dataset} \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
