#!/bin/bash
# build vocab for different datasets
setting=dataset/Biaffine/glove
python vocabulary/prepare_vocab.py --data_dir $setting/Restaurants --vocab_dir $setting/Restaurants
python vocabulary/prepare_vocab.py --data_dir $setting/Laptops --vocab_dir $setting/Laptops