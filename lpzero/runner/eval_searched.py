import argparse
import csv
import json

import logging
import math
import random
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from loguru import logger
from torch import Tensor
from transformers import ElectraTokenizerFast

from lpzero.model.flexibert.modeling_electra import (
    ElectraConfig,
    ElectraLayer,
    ElectraModel,
)
from lpzero.structures import GraphStructure, LinearStructure, TreeStructure
from lpzero.utils.rank_consistency import spearman
from lpzero.runner.evo_search import parse_args, all_same, generate_inputs, is_anomaly

configs = []
with open('./data/BERT_benchmark.json', 'r') as f:
    configs = json.load(f)

args = parse_args()



def fitness_spearman(inputs, structure, device=None, num_sample=50):
    """structure is belong to popultion."""
    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    if structure.sp_score != -1:
        return structure.sp_score

    gt_score = []
    zc_score = []

    for i in range(num_sample):
        nas_config = configs[i]['hparams']['model_hparam_overrides']['nas_config']

        gt = configs[i]['scores']['glue']
        # build a new model
        config = ElectraConfig(
            nas_config=nas_config,
            num_hidden_layers=len(nas_config['encoder_layers']),
            output_hidden_states=True,
        )
        model = ElectraModel(config)
        model.to(device)
        inputs.to(device)
        zc = structure(inputs, model)

        if is_anomaly(zc):
            return -1

        # early exit
        if len(zc_score) > 3 and all_same(zc_score):
            return -1

        zc_score.append(zc)
        gt_score.append(gt)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # gc.collect()

    # TODO add inf check
    try:
        if len(zc_score) <= 1 or np.isnan(spearman(gt_score, zc_score)):
            return -1
    except TypeError as e:
        import pdb

        pdb.set_trace()

    # release memory
    del inputs
    torch.cuda.empty_cache()
    # gc.collect()

    sp = spearman(gt_score, zc_score)
    if structure.sp_score == -1:
        structure.sp_score = sp
    return sp


if __name__ == '__main__':
    inputs = generate_inputs()

    # preprocess search space structure
    if args.search_structure == 'linear':
        structure = LinearStructure
    elif args.search_structure == 'tree':
        structure = TreeStructure
    elif args.search_structure == 'graph':
        structure = GraphStructure
    else:
        raise NotImplementedError(
            f'Not support {args.search_structure} structure.')

    print('Begin Evolution Search...')
    struct = structure()
    genotype = {
        'input_geno': ['head', 'act'],
        'op_geno': [[18, 3], [5, 12], 2]
    }
    struct.genotype = genotype
    # Struct=INPUT:(head, act)
    # TREE:(to_std_scalar-element_wise_pow|normalize-sigmoid)
    # BINARY:(element_wise_product) Input=['head', 'act'] Op=[[18, 3], [5, 12], 2]
    
    sp = fitness_spearman(inputs, struct, num_sample=args.num_sample)
    print(f"Spearman of {struct} is {sp}")
    
    