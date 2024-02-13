#!/bin/bash

#### EVOLUTION SEARCH for BERT 

# NUM_SAMPLE=50
# CUDA_VISIBLE_DEVICES=5 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_run6.log
    #  \
    # > logs/evo_search_run6.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_run2.log \
#     > logs/evo_search_run2.log 2>&1 &

# NUM_SAMPLE=100
# CUDA_VISIBLE_DEVICES=6 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_${NUM_SAMPLE}_run3.log \
#     > logs/evo_search_num_sample_${NUM_SAMPLE}_run3.log 2>&1 &

# NUM_SAMPLE=50
# CUDA_VISIBLE_DEVICES=7 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_${NUM_SAMPLE}_run4.log \
#     > logs/evo_search_${NUM_SAMPLE}_run4.log 2>&1 &

# NUM_SAMPLE=50
# STRUCTURE=graph
# CUDA_VISIBLE_DEVICES=2 python lpzero/runner/evo_search.py \
#     --search_structure ${STRUCTURE} \
#     --log_path ./logs/evo_search_${STRUCTURE}_RUN0.log
    # \
    # > ./logs/evo_search_${STRUCTURE}_run0.log 2>&1 &

# NUM_SAMPLE=100
# CUDA_VISIBLE_DEVICES=0 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_new_run0.log &

# NUM_SAMPLE=100
# CUDA_VISIBLE_DEVICES=1 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_new_run1.log &

# NUM_SAMPLE=100
# CUDA_VISIBLE_DEVICES=2 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_new_run2.log &

# NUM_SAMPLE=100
# CUDA_VISIBLE_DEVICES=3 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_new_run6.log &

# NUM_SAMPLE=100
# CUDA_VISIBLE_DEVICES=4 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_new_run7.log &

# NUM_SAMPLE=100
# CUDA_VISIBLE_DEVICES=5 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_new_run8.log &

# NUM_SAMPLE=100
# CUDA_VISIBLE_DEVICES=6 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_new_run9.log &

# NUM_SAMPLE=100
# CUDA_VISIBLE_DEVICES=7 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_new_run10.log &

# NUM_SAMPLE=500
# CUDA_VISIBLE_DEVICES=7 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_new_run10.log &

# NUM_SAMPLE=500
# CUDA_VISIBLE_DEVICES=6 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_new_run10.log &

#### EVOLUTION SEARCH for GPT2

NUM_SAMPLE=500 
CUDA_VISIBLE_DEVICES=0 python lpzero/runner/evo_search_gpt2.py \
    --log_path ./logs/evo_search_gpt2_run0.log &

NUM_SAMPLE=500
CUDA_VISIBLE_DEVICES=1 python lpzero/runner/evo_search_gpt2.py \
    --log_path ./logs/evo_search_gpt2_run1.log &

NUM_SAMPLE=500
CUDA_VISIBLE_DEVICES=2 python lpzero/runner/evo_search_gpt2.py \
    --log_path ./logs/evo_search_gpt2_run2.log &

NUM_SAMPLE=500
CUDA_VISIBLE_DEVICES=3 python lpzero/runner/evo_search_gpt2.py \
    --log_path ./logs/evo_search_gpt2_run3.log &

NUM_SAMPLE=500
CUDA_VISIBLE_DEVICES=4 python lpzero/runner/evo_search_gpt2.py \
    --log_path ./logs/evo_search_gpt2_run4.log &

NUM_SAMPLE=500
CUDA_VISIBLE_DEVICES=5 python lpzero/runner/evo_search_gpt2.py \
    --log_path ./logs/evo_search_gpt2_run5.log &

NUM_SAMPLE=500
CUDA_VISIBLE_DEVICES=6 python lpzero/runner/evo_search_gpt2.py \
    --log_path ./logs/evo_search_gpt2_run6.log &

NUM_SAMPLE=500
CUDA_VISIBLE_DEVICES=7 python lpzero/runner/evo_search_gpt2.py \
    --log_path ./logs/evo_search_gpt2_run7.log &


##### EVOLUTION SEARCH for BERT (ablation study)

# NUNARY=2
# CUDA_VISIBLE_DEVICES=0 python lpzero/runner/evo_search_bert.py --n_unary $NUNARY \
#     --log_path ./logs/evo_search_ablation_unary_number_NUNARY_${NUNARY}_run0.log 

# NUNARY=3
# CUDA_VISIBLE_DEVICES=1 python lpzero/runner/evo_search_bert.py --n_unary $NUNARY \
#     --log_path ./logs/evo_search_ablation_unary_number_NUNARY_${NUNARY}_run1.log &

# NUNARY=4
# CUDA_VISIBLE_DEVICES=2 python lpzero/runner/evo_search_bert.py --n_unary $NUNARY \
#     --log_path ./logs/evo_search_ablation_unary_number_NUNARY_${NUNARY}_run2.log &

# NUNARY=5
# CUDA_VISIBLE_DEVICES=3 python lpzero/runner/evo_search_bert.py --n_unary $NUNARY \
#     --log_path ./logs/evo_search_ablation_unary_number_NUNARY_${NUNARY}_run3.log &
