#!/bin/bash 

n_gpus=4 
model_type="hf_gpt2_flex"
gpu_config="dgx1_8gpu_fp32"
default_config_file="lpzero/nlp/wt103_base.yaml"
dataset="wt103"
max_step=1000
vocab="gpt2"
vocab_size=50257
exp_name="gpt2_wikitext2"
data_root=/mnt/sdb/xxx/workspace/LPZero/data/wikitext/


python -m torch.distributed.launch --nproc_per_node=${n_gpus} lpzero/runner/train.py --model_type ${model_type} --config ${gpu_config} \
        --config_file ${default_config_file} --dataset ${dataset} --max_step ${max_step} --vocab ${vocab} --data ${data_root} \
        --vocab_size ${vocab_size} --experiment_name ${exp_name} ${config_line}

# /mnt/sdb/xxx/workspace/LPZero/data/wikitext/wikitext-103/wiki.train.tokens