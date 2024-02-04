#!/bin/bash

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

# NUM_SAMPLE=200
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

# refactor denotes the operations are more diverse
NUM_SAMPLE=200
CUDA_VISIBLE_DEVICES=5 python lpzero/runner/evo_search.py \
    --log_path ./logs/evo_search_cci_diversity_refactor_run6.log
   # \
   # > logs/evo_search_cci_diversity_refactor_run6.log 2>&1 &

# NUM_SAMPLE=200
# CUDA_VISIBLE_DEVICES=6 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_cci_diversity_refactor_run7.log \
#     > logs/evo_search_cci_diversity_refactor_run7.log 2>&1 &

# NUM_SAMPLE=200
# CUDA_VISIBLE_DEVICES=4 python lpzero/runner/evo_search.py \
#     --log_path ./logs/evo_search_cci_diversity_refactor_run8.log \
#     > logs/evo_search_cci_diversity_refactor_run8.log 2>&1 &
