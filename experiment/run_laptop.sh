#!/usr/bin/env bash
cd ..

now_cu=$1
seed=33
addition=best
size=10k

source_dir=dataset
save_dir=experiment/log

exp_dataset=Biaffine/glove/Laptops

############# Restaurants acc:86.68 f1:80.92 #################

exp_setting=$seed

exp_dir=$save_dir/Laptops/$addition
if [ ! -d "$exp_dir" ]; then
  mkdir -p "$exp_dir"
fi

exp_path=$exp_dir/$exp_setting
if [ ! -d "$exp_path" ]; then
  mkdir -p "$exp_path"
fi

CUDA_VISIBLE_DEVICES=$now_cu python3 -u main.py \
	--en_lr 5e-4 \
	--de_lr 1e-4 \
	--bert_lr 1e-5 \
	--anti_weight 1e-5 \
	--sentence_weight 0.1 \
	--pseudo_weight 0.5 \
	--bert_out_dim 256 \
	--input_dropout 0.3 \
	--att_dropout 0.2 \
	--l2 0 \
	--aux_round 1 \
	--auto_lr 1 \
	--num_layer 6 \
	--batch_size 8 \
	--dep_dim 100 \
	--max_len 90 \
	--data_dir $source_dir/$exp_dataset \
	--vocab_dir $source_dir/$exp_dataset \
	--save_dir $exp_path \
	--model "RGAT" \
	--output_merge "gate" \
	--reset_pool \
	--seed $seed \
	--dataset_size laptop_$size \
	--num_epoch 30 \
	--decay_patience 3 \
	--bert_from "pre_model/laptop" \
	--step_long 2 2>&1 | tee $exp_path/$exp_setting.log

cd -
