#!/bin/bash 


#num_parameters
# METHOD=num_parameters
# CUDA_VISIBLE_DEVICES=7 python lpzero/runner/eval_rank_gpt2.py \
#     --method $METHOD \
#     --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
#      > ./logs/rank_corr_${METHOD}.log 2>&1 &

# lpzero 
METHOD=lpzero
CUDA_VISIBLE_DEVICES=2 python lpzero/runner/eval_rank_gpt2.py \
    --method $METHOD \
    --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
     > ./logs/rank_corr_${METHOD}_aftersearch.log 2>&1 &

#synaptic_diversity 
# METHOD=synaptic_diversity
# CUDA_VISIBLE_DEVICES=6 python lpzero/runner/eval_rank_gpt2.py \
#     --method $METHOD \
#     --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
#      > ./logs/rank_corr_${METHOD}.log 2>&1 &

#synaptic_saliency 
# METHOD=synaptic_saliency
# CUDA_VISIBLE_DEVICES=5 python lpzero/runner/eval_rank_gpt2.py \
#     --method $METHOD \
#     --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
#      > ./logs/rank_corr_${METHOD}.log 2>&1 &

# activation_distance 
# METHOD=activation_distance
# CUDA_VISIBLE_DEVICES=4 python lpzero/runner/eval_rank_gpt2.py \
#     --method $METHOD \
#     --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
#      > ./logs/rank_corr_${METHOD}.log 2>&1 &

# jacobian_cosine
# METHOD=jacobian_cosine
# CUDA_VISIBLE_DEVICES=3 python lpzero/runner/eval_rank_gpt2.py \
#     --method $METHOD \
#     --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
#      >> ./logs/rank_corr_${METHOD}.log 2>&1 &


# attention_confidence 
# METHOD=attention_confidence
# CUDA_VISIBLE_DEVICES=1 python lpzero/runner/eval_rank_gpt2.py \
#     --method $METHOD \
#     --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
#      > ./logs/rank_corr_${METHOD}.log 2>&1 &

# attention importance 
# METHOD=attention_importance
# CUDA_VISIBLE_DEVICES=0 python lpzero/runner/eval_rank_gpt2.py \
#     --method $METHOD \
#     --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost 
    # \
    #  > ./logs/rank_corr_${METHOD}.log 2>&1 &

# head_importance 
# METHOD=head_importance
# CUDA_VISIBLE_DEVICES=0 python lpzero/runner/eval_rank_gpt2.py \
#     --method $METHOD \
#     --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
#      > ./logs/rank_corr_${METHOD}.log 2>&1 &

# head_confidence
# METHOD=head_confidence
# CUDA_VISIBLE_DEVICES=6 python lpzero/runner/eval_rank_gpt2.py \
#     --method $METHOD \
#     --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
#      > ./logs/rank_corr_${METHOD}.log 2>&1 &

# eznas 
# METHOD=eznas
# CUDA_VISIBLE_DEVICES=7 python lpzero/runner/eval_rank_gpt2.py \
#     --method $METHOD \
#     --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
#      > ./logs/rank_corr_${METHOD}.log 2>&1 &

# logsynflow 
# METHOD=logsynflow
# CUDA_VISIBLE_DEVICES=0 python lpzero/runner/eval_rank_gpt2.py \
#     --method $METHOD \
#     --exp_name ./saved_logs/random_GPT2_wt103 --plot --get_cost \
#      > ./logs/rank_corr_${METHOD}.log 2>&1 &