import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from lpzero.model.flexibert.modeling_electra import (
    ElectraConfig,
    ElectraModel,
)
from lpzero.runner.evo_search_bert import all_same, generate_inputs, is_anomaly, parse_args
from lpzero.structures import GraphStructure, LinearStructure, TreeStructure
from lpzero.utils.rank_consistency import spearman, kendalltau
from lpzero.predictor.measures.fisher import compute_fisher_per_weight
from lpzero.predictor.measures.snip import compute_snip_per_weight 
from lpzero.predictor.measures.grasp import compute_grasp_per_weight
from lpzero.predictor.measures.synflow import compute_synflow_per_weight, compute_logsynflow_per_weight
from lpzero.predictor.measures.grad_norm import get_grad_norm_arr
from lpzero.predictor.measures.activation_distance import compute_act_dist
from lpzero.predictor.measures.synaptic_diversity import synaptic_diversity
from lpzero.predictor.measures.synaptic_saliency import compute_synaptic_saliency
from lpzero.predictor.measures.jacobian_score import jacobian_score_cosine
from lpzero.predictor.measures.attention_confidence import compute_attention_confidence

configs = []
with open('./data/BERT_benchmark.json', 'r') as f:
    configs = json.load(f)

args = parse_args()


def find_best_w_structure(inputs, structure, device=None, num_sample=500):
    """structure is belong to popultion."""
    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    best_gt = -1 
    best_zc = -1
    best_nas_config = None
    best_scores = None 
    
    for i in tqdm(range(num_sample)):
        nas_config = configs[i]['hparams']['model_hparam_overrides']['nas_config']

        gt = configs[i]['scores']['glue']
        _score = configs[i]['scores']
        
        # build a new model
        config = ElectraConfig(
            nas_config=nas_config,
            num_hidden_layers=len(nas_config['encoder_layers']),
            output_hidden_states=True,
        )
        model = ElectraModel(config)
        model.to(device)
        inputs.to(device)
        
        zc = structure(model, inputs)
        
        if best_zc < zc:
            best_zc = zc
            best_gt = gt
            best_nas_config = nas_config
            best_scores = _score
    
    return best_gt, best_nas_config, best_scores


def sum_arr(arr):
    if isinstance(arr, torch.Tensor):
        return torch.sum(arr).item()
    if isinstance(arr, (int, float)):
        return arr
    if isinstance(arr[0], (float, int)):
        return sum(arr)
    
    _sum = 0.0
    for i in range(len(arr)):
        _sum += torch.sum(arr[i])
    return _sum.item() 

def find_best_w_predictive(inputs, proxy_type, device=None, num_sample=500):
    device = device or torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    
    best_gt = -1 
    best_zc = -1
    best_nas_config = None
    best_scores = None 
    
    for i in tqdm(range(num_sample)):
        nas_config = configs[i]['hparams']['model_hparam_overrides']['nas_config']

        gt = configs[i]['scores']['glue']
        _score = configs[i]['scores']
        
        # build a new model
        config = ElectraConfig(
            nas_config=nas_config,
            num_hidden_layers=len(nas_config['encoder_layers']),
            output_hidden_states=True,
        )
        model = ElectraModel(config)
        
        model.to(device)
        
        if isinstance(inputs, torch.Tensor):
            inputs.to(device)
        else:
            inputs = {k: v.to(device) for k, v in inputs.items()}        
        
        if proxy_type == 'fisher':
            zc = compute_fisher_per_weight(model, inputs)
        elif proxy_type == 'snip':
            zc = compute_snip_per_weight(model, inputs)
        elif proxy_type == 'grasp':
            zc = compute_grasp_per_weight(model, inputs)
        elif proxy_type == 'synflow':
            zc = compute_synflow_per_weight(model, inputs)
        elif proxy_type == 'logsynflow':
            zc = compute_logsynflow_per_weight(model, inputs)
        elif proxy_type == 'gradnorm':
            zc = get_grad_norm_arr(model, inputs)
        elif proxy_type == 'act_dist':
            zc = compute_act_dist(model, inputs)
        elif proxy_type == 'syn_div':
            zc = synaptic_diversity(model)
        elif proxy_type == 'syn_sal':
            zc = compute_synaptic_saliency(model, inputs)
        elif proxy_type == 'jacob_cos':
            zc = jacobian_score_cosine(model, inputs)
        elif proxy_type == 'att_conf':
            zc = compute_attention_confidence(model, inputs)
        else: 
            raise NotImplementedError(f'Not support {proxy_type} proxy.')
              
        if isinstance(zc, list):
            zc = zc[0]
            
        zc = sum_arr(zc)    
        
        if best_zc < zc:
            best_zc = zc
            best_gt = gt
            best_nas_config = nas_config
            best_scores = _score
    
    return best_gt, best_nas_config, best_scores


if __name__ == '__main__':
    inputs = generate_inputs()

    if args.type == 'lpzero':

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
        # genotype = {'input_geno': ['head', 'act'],
        #             'op_geno': [[18, 3], [5, 12], 2]} # 0.7581
        # genotype = {
        #     'input_geno': ['jacobs', 'head'],
        #     'op_geno': [[14, 8], [15, 10], 0],
        # }  # 0.7474
        # genotype = { # better
        #     'input_geno': ['head', 'act'],
        #     'op_geno': [[2, 17], [18, 13], 0],
        # }
        # genotype = { # 0.6928 but better
        #     'input_geno': ['head', 'head'],
        #     'op_geno': [[3, 11], [8, 19], 0],
        # }
        # genotype = { # 0.6803 but better
        #     'input_geno': ['head', 'weight'],
        #     'op_geno': [[8, 19], [17, 8], 0],
        # }
        # genotype = { # 0.6917 but better
        #     'input_geno': ['softmax', 'head'],
        #     'op_geno': [[19, 14], [8, 6], 0],
        # }
        # 2024-02-06 21:08:17 | INFO | evo_search:evolution_search:248 - ====> 
        # After 1000 iters: Best SP:0.8515246098439375 Struct=INPUT:(head, act)TREE:(element_wise_revert|element_wise_pow|frobenius_norm|logsoftmax)BINARY:(element_wise_sum) 
        # Input=['head', 'act'] Op=[[15, 3], [8, 13], 0]

        genotype = {
            'input_geno': ['head', 'act'],
            'op_geno': [[15, 3], [8, 13], 0],
        }

        struct.genotype = genotype
        # Struct=INPUT:(head, act)
        # TREE:(to_std_scalar-element_wise_pow|normalize-sigmoid)
        # BINARY:(element_wise_product) Input=['head', 'act'] Op=[[18, 3], [5, 12], 2]

        gt, nas_config, best_score = find_best_w_structure(inputs, struct)
    
    else: 
        gt, nas_config, best_score = find_best_w_predictive(inputs, proxy_type=args.type)
        
        

    print(f'Ground Truth: {gt}')
    print(f'NAS Config: {nas_config}')
    print(f'Best Score: {best_score}')