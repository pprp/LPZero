#!/bin/bash 

CUDA_VISIBLE_DEVICES=0 python lpzero/runner/evo_search.py \
    > logs/evo_search_run0.log 2>&1 &