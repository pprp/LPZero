#!/bin/bash 

# # fisher 
# CUDA_VISIBLE_DEVICES=0 python lpzero/runner/find_best_bert.py --type fisher > ./logs/find_best/fisher.log 2>&1 &

# # snip
# CUDA_VISIBLE_DEVICES=1 python lpzero/runner/find_best_bert.py --type snip > ./logs/find_best/snip.log 2>&1 & 

# # grasp
# CUDA_VISIBLE_DEVICES=2 python lpzero/runner/find_best_bert.py --type grasp > ./logs/find_best/grasp.log 2>&1 &

# # synflow
# CUDA_VISIBLE_DEVICES=3 python lpzero/runner/find_best_bert.py --type synflow > ./logs/find_best/synflow.log 2>&1 &

# # logsynflow 
# CUDA_VISIBLE_DEVICES=4 python lpzero/runner/find_best_bert.py --type logsynflow > ./logs/find_best/logsynflow.log 2>&1 &

# # gradnorm 
# CUDA_VISIBLE_DEVICES=5 python lpzero/runner/find_best_bert.py --type gradnorm > ./logs/find_best/gradnorm.log 2>&1 &

# lpzero 
# CUDA_VISIBLE_DEVICES=6 python lpzero/runner/find_best_bert.py --type lpzero > ./logs/find_best/lpzero.log 2>&1 &

# act_dist 
CUDA_VISIBLE_DEVICES=7 python lpzero/runner/find_best_bert.py --type act_dist > ./logs/find_best/act_dist.log 2>&1 &
tail -f ./logs/find_best/act_dist.log

# # syn_div 
# CUDA_VISIBLE_DEVICES=0 python lpzero/runner/find_best_bert.py --type syn_div > ./logs/find_best/syn_div.log 2>&1 &

# # syn_sal
# CUDA_VISIBLE_DEVICES=1 python lpzero/runner/find_best_bert.py --type syn_sal > ./logs/find_best/syn_sal.log 2>&1 &

# # jacob_cos
# CUDA_VISIBLE_DEVICES=2 python lpzero/runner/find_best_bert.py --type jacob_cos > ./logs/find_best/jacob_cos.log 2>&1 &

# att_conf
# CUDA_VISIBLE_DEVICES=3 python lpzero/runner/find_best_bert.py --type att_conf > ./logs/find_best/att_conf.log 2>&1 &
# tail -f ./logs/find_best/att_conf.log

# head_imp 
# CUDA_VISIBLE_DEVICES=4 python lpzero/runner/find_best_bert.py --type head_imp > ./logs/find_best/head_imp.log 2>&1 &

# softmax_conf
# CUDA_VISIBLE_DEVICES=5 python lpzero/runner/find_best_bert.py --type softmax_conf > ./logs/find_best/softmax_conf.log 2>&1 &
# tail -f ./logs/find_best/softmax_conf.log

# params 
# CUDA_VISIBLE_DEVICES=6 python lpzero/runner/find_best_bert.py --type params > ./logs/find_best/params.log 2>&1 &
# tail -f ./logs/find_best/params.log