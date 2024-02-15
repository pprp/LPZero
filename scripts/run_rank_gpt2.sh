#!/bin/bash 


CUDA_VISIBLE_DEVICES=7 python lpzero/runner/eval_rank_gpt2.py \
    --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost
     > ./logs/rank_corr.log

# tail -f rank_corr.log
